import json

from datasets import Dataset, DatasetDict

verification_result_file = "xuhao/verify/data/output/verification_result_claude.json"
verification_result_label_file = "xuhao/verify/data/output/verification_result_label_claude.json"
verification_data_file = "xuhao/verify/data/input/verification_data.json"
sft_dataset_file = "xuhao/sft/data/input/sft_data.json"

def get_verification_input(verification_data, problem_index, step_index):
    for data in verification_data:
        if data["problem_index"] == problem_index and data["step_index"] == step_index:
            return data["verification_input"]

def prepare_sft_dataset(train_test_ratio=24):
    with open(verification_result_file, 'r') as f1, open(verification_data_file, 'r') as f2:
        verification_result = json.load(f1)
        verification_data = json.load(f2)
    with open(verification_result_label_file, 'r') as f:
        verification_result_label = json.load(f)

    data = []
    for i in range(len(verification_result)):
        if verification_result_label[verification_result[i]["problem_index"]]["verification_result_label"]:
            # verification to this problem is right
            problem_index = verification_result[i]["problem_index"]
            step_index = verification_result[i]["step_index"]
            data.append({
                "problem_index": problem_index,
                "step_index": step_index,
                "input": get_verification_input(verification_data, problem_index, step_index),
                "output": verification_result[i]["verification_result"]
            })
    
    num_train_data = int(len(data) * (train_test_ratio / (train_test_ratio + 1)))

    # find the problem-problem boundary
    new_num_train_data = num_train_data
    for i in range(num_train_data, len(data)):
        if data[i]["problem_index"] == data[num_train_data - 1]["problem_index"]:
            new_num_train_data += 1
    
    num_train_data = new_num_train_data
    num_test_data = len(data) - num_train_data
    total = len(data)

    final_data = {
        "train": data[:num_train_data],
        "test": data[num_train_data:]
    }

    print(f"Total Data: {total}")
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
    dataset_dict.save_to_disk('xuhao/sft/data/input/sft_data')
    
    return dataset_dict

if __name__ == "__main__":
    prepare_sft_dataset(train_test_ratio=9)
    convert_dataset_format()
