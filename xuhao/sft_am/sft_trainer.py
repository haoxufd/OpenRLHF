import json
import os
from abc import ABC

import torch
import torch.distributed as dist
from flash_attn.utils.distributed import all_gather
from torch.optim import Optimizer
from torch.nn import functional as F
from tqdm import tqdm

from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class SFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        eval_gen_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        eval_gen_tokenizer=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_gen_dataloader = eval_gen_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.eval_gen_tokenizer = eval_gen_tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for prompt_id_lens, inputs, attention_masks, infos in self.train_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs, 
                        attention_mask=attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                gpt_loss = self.loss_fn(output.logits, labels)
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
        
        def get_next_eval_gen_file(directory='xuhao/sft_am/data/output'):
            import re
            # 获取目录下所有文件
            files = os.listdir(directory)
            
            # 用正则表达式匹配 eval_step_数字.json 格式的文件
            pattern = re.compile(r'eval_step_(\d+)\.json$')
            
            # 找出所有匹配的文件，提取编号
            step_numbers = []
            for file in files:
                match = pattern.match(file)
                if match:
                    step_numbers.append(int(match.group(1)))
            
            # 如果没有找到匹配的文件，返回 0
            if not step_numbers:
                return "xuhao/sft_am/data/output/eval_step_0.json"
            
            # 否则返回最大编号 + 1
            return "xuhao/sft_am/data/output/eval_step_{}.json".format(max(step_numbers)+1)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        
        # compute verification accuracy in eval dataset
        if global_step % args.eval_acc_steps == 0:
            if len(self.eval_dataloader) > 0:
                generated_results = self.generate(self.eval_gen_dataloader, global_step)
                if self.strategy.is_rank_0():
                    with open(get_next_eval_gen_file(), 'w') as f:
                        json.dump(generated_results, f, indent=4)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def generate(self, eval_gen_dataloader, steps=0):
        """生成文本的独立函数
        
        Args:
            eval_dataloader: 评估数据集的dataloader
            steps: 当前的训练步数
            
        Returns:
            generated_results: 按原始数据集顺序排列的生成结果列表
        """
        self.model.eval()
        
        indexed_results = []
        
        pbar = tqdm(
            eval_gen_dataloader,
            desc="Generate stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )

        def tokenize_fn(texts):
            assert self.eval_gen_tokenizer is not None
            batch = self.eval_gen_tokenizer(
                texts,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=2048,
                padding=True,
                truncation=True,
            )
            return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

        with torch.no_grad():
            for micro_batch_idx, prompts in enumerate(pbar):
                # 获取当前batch的全局索引
                batch_size = len(prompts)
                rank = dist.get_rank() if dist.is_initialized() else 0
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                base_idx = micro_batch_idx * self.args.micro_train_batch_size * world_size + rank
                batch_indices = [base_idx + i * world_size for i in range(batch_size)]
                

                inputs = tokenize_fn(prompts)

                # 生成文本
                generate_kwargs = {
                    "max_new_tokens": 1024,  # 可配置参数
                    "do_sample": True,
                    "top_p": 1.0,
                    "temperature": 1.0,
                    "early_stopping": False,
                    "num_beams": 2,
                    "pad_token_id": self.eval_gen_tokenizer.pad_token_id,
                    "eos_token_id": self.eval_gen_tokenizer.eos_token_id,
                }
                generated_tokens = self.model.model.generate(
                    **inputs,
                    **generate_kwargs
                )

                input_length = inputs["input_ids"].shape[1]
                responses = self.eval_gen_tokenizer.batch_decode(generated_tokens[:, input_length:], skip_special_tokens=True)

                for response, global_idx in zip(responses, batch_indices):
                    indexed_results.append((global_idx, response))

        # 收集所有GPU的结果
        if dist.is_initialized():
            gathered_results = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_results, indexed_results)
            # 展平并按原始索引排序
            all_results = []
            for process_results in gathered_results:
                all_results.extend(process_results)
            all_results.sort(key=lambda x: x[0])  # 按索引排序
            generated_results = [text for _, text in all_results]  # 只保留文本部分
        else:
            # 非分布式情况
            indexed_results.sort(key=lambda x: x[0])
            generated_results = [text for _, text in indexed_results]
        
        self.model.train()
        return generated_results

    def evaluate(self, eval_dataloader, steps=0):
        """评估函数: 计算loss并生成文本"""
        times = 0
        self.model.eval()
        
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            # 计算loss
            for prompt_id_lens, inputs, attention_masks, infos in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs, 
                        attention_mask=attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )
                    
                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)
                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
               
        self.model.train()  # reset model state