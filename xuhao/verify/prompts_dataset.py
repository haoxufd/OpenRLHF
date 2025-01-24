from torch.utils.data import Dataset
from tqdm import tqdm
import json


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    with open("/root/OpenRLHF/xuhao/verification_system_message.txt", 'r') as f:
        verification_system_message = f.read()
    
    with open("/root/OpenRLHF/xuhao/verification_few_shot.json", 'r') as f:
        few_shot_examples = json.load(f)

    data = data[input_key]
    if apply_chat_template:
        if isinstance(data, str):
            chat = [{"role": "system", "content": verification_system_message}]
            for example in few_shot_examples:
                chat.append({"role": "user", "content": example["input"]})
                chat.append({"role": "assistant", "content": example["output"]})
            chat.append({"role": "user", "content": data})
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            return prompt
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        return prompt


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
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]