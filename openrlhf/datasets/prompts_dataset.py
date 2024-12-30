from torch.utils.data import Dataset
from tqdm import tqdm
import json


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    with open("/home/user/OpenRLHF/xuhao/solution_system_message.txt", 'r') as f:
        solution_system_message = f.read()
    
    with open("/home/user/OpenRLHF/xuhao/solution_few_shot.json", 'r') as f:
        few_shot = json.load(f)

    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "system", "content": solution_system_message}]
            for fs in few_shot:
                chat.append({"role": "user", "content": fs["question"]})
                chat.append({"role": "assistant", "content": fs["answer"]})
            chat.append({"role": "user", "content": data[input_key]})
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
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
            self.prompts.append((prompt, data["answer"], data["question"]))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
