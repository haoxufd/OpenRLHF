import argparse
import os
from datetime import timedelta

import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm

from openrlhf.models import Actor
from openrlhf.utils import get_strategy, get_tokenizer

import json
from torch.utils.data import Dataset

from xuhao.utils import blending_datasets

home_dir = "/root"
solution_system_message_file = f"{home_dir}/OpenRLHF/xuhao/sft_am/data/input/system_message.txt"
solution_few_shot_file = f"{home_dir}/OpenRLHF/xuhao/sft_am/data/input/few_shot.json"

def preprocess_data(data, input_key, apply_chat_template) -> str:
    with open(solution_system_message_file, 'r') as f1, open(solution_few_shot_file, 'r') as f2:
        system_message = f1.read()
        few_shot_examples = json.load(f2)

    data = data[input_key]
    messages = [{"role": "system", "content": system_message}]
    for example in few_shot_examples:
        messages.append({"role": "user", "content": example["problem"]})
        messages.append({"role": "assistant", "content": example["solution"]})
    messages.append({"role": "user", "content": data})
    prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

class PromptDatasetEval(Dataset):
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