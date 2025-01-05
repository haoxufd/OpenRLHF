import json

from datasets import Dataset, DatasetDict

claude_verification_result_file = "xuhao/claude_verification_result_small.json"
verification_dataset_file = "xuhao/verification_dataset.json"
sft_dataset_file = "xuhao/sft/sft_data.json"

def prepare_sft_dataset(train_test_ratio=9):
    data = []
    with open(claude_verification_result_file, 'r') as f:
        claude_verification_result = json.load(f)
    with open(verification_dataset_file, 'r') as f:
        verification_dataset = json.load(f)
    
    for i, verifications_for_a_problem in enumerate(claude_verification_result):
        for j, verification in enumerate(verifications_for_a_problem):
            data.append({
                "problem idx": i,
                "input": verification_dataset[i]["input"][j],
                "output": verification
            })
            if verification.strip().lower().startswith("evaluation: incorrect"):
                break
    
    num_train_data = int(len(data) * (train_test_ratio / (train_test_ratio + 1)))
    num_test_data = len(data) - num_train_data
    
    final_data = {
        "train": data[:num_train_data],
        "test": data[num_train_data:]
    }

    print(f"Num Train Data: {num_train_data}")
    print(f"Num Test Data: {num_test_data}")
    
    with open(sft_dataset_file, 'w') as f:
        json.dump(final_data, f, indent=4)

def convert_dataset_format():
    # 读取原始数据
    with open(sft_dataset_file, 'r') as f:
        data = json.load(f)

    # 假设列表中每个元素是字典，我们需要重组数据
    # 将列表转换为按列组织的字典
    train_dict = {}
    if data['train']:  # 确保列表不为空
        # 获取所有键
        keys = data['train'][0].keys()
        # 按列重组数据
        for key in keys:
            train_dict[key] = [item[key] for item in data['train']]

    test_dict = {}
    if data['test']:  # 确保列表不为空
        keys = data['test'][0].keys()
        for key in keys:
            test_dict[key] = [item[key] for item in data['test']]

    # 创建 DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict(train_dict),
        'test': Dataset.from_dict(test_dict)
    })

    # 保存到磁盘
    dataset_dict.save_to_disk('xuhao/sft/data')
    
    return dataset_dict

# 调用函数
dataset = convert_dataset_format()

if __name__ == "__main__":
    convert_dataset_format()
