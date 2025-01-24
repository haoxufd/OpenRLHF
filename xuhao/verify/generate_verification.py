import argparse
import os
from datetime import timedelta
from tkinter import NO

import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer

import json
from torch.utils.data import Dataset

from xuhao.utils import read_json_list_file, write_json_list_file

verification_system_message_file = "/root/OpenRLHF/xuhao/verify/data/input/verification_system_message.txt"
verification_few_shot_file = "/root/OpenRLHF/xuhao/verify/data/input/verification_few_shot.json"

def preprocess_data(data, input_key, apply_chat_template) -> dict:
    with open(verification_system_message_file, 'r') as f1, open(verification_few_shot_file, 'r') as f2:
        system_message = f1.read()
        few_shot_examples = json.load(f2)

    messages = [{"role": "system", "content": system_message}]
    for example in few_shot_examples:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})
    messages.append({"role": "user", "content": data[input_key]})
    prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return {
        "problem_index": data["problem_index"],
        "step_index": data["step_index"],
        "prompt": prompt
    }

class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]

def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        desc="Generating",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    indexed_outputs = []

    for batch_idx, prompts in enumerate(pbar):
        prompt_contents = prompts["prompt"]
        inputs = tokenize_fn(prompt_contents)
        for _ in range(N):
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=False,
                num_beams=2,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_length = inputs["input_ids"].shape[1]
            outputs = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=False)
            for i, output in enumerate(outputs):
                # 保存索引和输出的对应关系
                indexed_outputs.append({
                    "problem_index": int(prompts["problem_index"][i]),
                    "step_index": int(prompts["step_index"][i]),
                    "verification_result": output
                })
    
    write_json_list_file(args.output_path + str(strategy.get_rank()), indexed_outputs)

    dist.barrier()

    # 在 rank 0 进程中合并并排序结果
    if strategy.is_rank_0():
        all_outputs = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        
        # 收集所有结果
        for file in files:
            all_outputs.extend(read_json_list_file(file))
            os.remove(file)
        
        # 按原始索引排序
        all_outputs.sort(key=lambda x: (x["problem_index"], x["step_index"]))
        
        with open(args.output_path, 'w') as f:
            json.dump(all_outputs[:len(prompts_data)], f, indent=4)

if __name__ == "__main__":
    pretrain = "/mnt/data/user/zhao_jun/xuhao/reward-llama-3.1-8b-sft-gsm8k"
    dataset = "xuhao/verify/data/input/verification_data_sft_test_new.json"
    input_key = "verification_input"
    max_samples = 1e8
    output_path = "xuhao/verify/data/output/verification_result_sft_test_new_ckpt2.json"
    prompt_max_length = 4096
    max_new_tokens = 1024
    micro_batch_size = 8
    train_batch_size = 32

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_task", type=str, default="generate", help="Set to generate_vllm, generate (HF generate) or rm"
    )
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=True, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAtten2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=micro_batch_size)
    parser.add_argument("--train_batch_size", type=int, default=train_batch_size)
    parser.add_argument("--seed", type=int, default=1234)

    # Models
    parser.add_argument("--pretrain", type=str, default=pretrain, help="HF pretrain model name or path")
    parser.add_argument(
        "--value_head_prefix", type=str, default="value_head", help="value_head prefix for Reward Model"
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=dataset)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default=input_key, help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=True, help="HF tokenizer apply_chat_template"
    )
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max_samples", type=int, default=max_samples, help="Max number of samples")
    parser.add_argument("--output_path", type=str, default=output_path, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=prompt_max_length, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=max_new_tokens, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), csft (Conditional SFT), iter_dpo (Iterative DPO) or None",
    )
    # For vllm
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)

    # For Iterative generation and Rejection Sampling
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=2048, help="Number of samples to generate")

    # For Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")

    args = parser.parse_args()

    batch_generate(args=args)
