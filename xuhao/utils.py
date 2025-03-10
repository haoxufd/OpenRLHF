import json
from itertools import groupby
from datasets import interleave_datasets, load_dataset
import re
from deepspeed import get_accelerator
from sympy import denom
import torch

def print_item_num(json_file):
    with open(json_file, 'r') as f:
        print(len(json.load(f)))

def read_json_list_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def write_json_list_file(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def get_redistill_problem(solution_file, old_solution_file):
    """
    获取需要重蒸馏的数据, solution 中跟 old_solution 重复的不需要重新蒸馏, 反正需要重新蒸馏
    返回一个 hash table, 长度为 problem 总数, 置为 True 的是需要重新蒸馏的
    """
    solution = read_json_list_file(solution_file)
    old_solution = read_json_list_file(old_solution_file)
    assert len(solution) == len(old_solution)

    num = len(solution)
    ret = [False] * num
    cnt = 0
    for i in range(num):
        if solution[i]["solution"] != old_solution[i]["solution"]:
            ret[i] = True
            cnt += 1
    print(cnt)
    return ret

def get_grouped_data(data):
    """
    convert format like this:
    [
        {"problem_index": 0, "step_index": 0},
        {"problem_index": 0, "step_index": 1},
        {"problem_index": 1, "step_index": 0},
        {"problem_index": 1, "step_index": 1}
    ]
    to this:
    [
        [
            {"problem_index": 0, "step_index": 0},
            {"problem_index": 0, "step_index": 1}
        ],
        [
            {"problem_index": 1, "step_index": 0},
            {"problem_index": 1, "step_index": 1}
        ]
    ]
    """
    return [list(group) for key, group in groupby(data, key=lambda x: x["problem_index"])]

def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        
        data = load_dataset(dataset, "main")
        strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)
        
        # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset

def parse_solution(solution: str):
    value = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', solution)
    value = value[0].replace(',', '') if len(value) > 0 else None
    return float(value) if value is not None else None

def get_solve_result(solutions: list, ref_solutions: list):
    """
    由于是通过 batch inference 得到的 solution, 其中的数据可能多于 ref_solution
    但两者数据是按顺序对应的, 遍历的时候以 ref_solution 为主即可
    """
    num_correct = num_incorrect = 0
    incorrect_indices = []
    for i in range(len(ref_solutions)):
        value = parse_solution(solutions[i])
        ref_value = parse_solution(ref_solutions[i])

        if value is None:
            num_incorrect += 1
            incorrect_indices.append(i)
        elif value is not None and value != ref_value:
            num_incorrect += 1
            incorrect_indices.append(i)
        else:
            num_correct += 1

    total = num_correct + num_incorrect
    return [num_correct, num_incorrect, total, num_correct / total, incorrect_indices]

def get_steps(solution: str):
    """
    solution 的格式一定是正确的
    """
    steps = solution.strip().split('<|reserved_special_token_0|>')
    # steps[-1] = <|eot_id|>
    steps = steps[:-1]
    # steps[-1] 以 #### FINAL_VALUE 结束, 需要去掉这个后缀, 因为 reward model 训练的时候最后一步没带后缀
    steps[-1] = steps[-1].split('\n')[0]
    # 去掉 step 最后的 \n, 同样是因为 reward model 训练的时候 step 没有以 \n 结束
    steps = [step.strip() for step in steps]
    return steps

def str_to_num(num_str: str):
    if '/' not in num_str:
        return float(num_str.replace(',', ''))
    else:
        numerator = float(num_str.split('/')[0].replace(',', ''))
        denominator = float(num_str.split('/')[1].replace(',', ''))
        if denominator == 0:
            return None
        return numerator / denominator

def get_final_value_from_solution(solution: str) -> float | None:
    """
    solution 的格式一定是正确的
    """
    end_mark = "<|reserved_special_token_0|><|eot_id|>"
    number_content = solution.split("#### ")[-1][:-len(end_mark)]
    assert ' ' not in number_content
    return str_to_num(number_content)

def get_final_value_from_ref_solution(ref_solution: str) -> float:
    number_content = ref_solution.split("#### ")[-1]
    return float(number_content)

def find_newline_indices(s):
    indices = []
    for i, char in enumerate(s):
        if char == '\n':
            # 如果是第一个 '\n' 或与前一个字符不连续
            if i == 0 or s[i - 1] != '\n':
                indices.append(i)
    return indices

def is_number(text: str) -> bool:
    pattern = r'^-?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\d+/\d+|\.\d+)$'
    match = re.fullmatch(pattern, text)
    return match is not None

def group_elements(elements, group_sizes):
    """
    将列表 elements 中的元素按照 group_sizes 中指定的分组大小分组.

    参数:
        elements (list): 待分组的元素列表, 其长度必须等于 sum(group_sizes).
        group_sizes (list of int): 每组需要的元素个数列表, 总和应与 elements 的长度相等.

    返回:
        list of list: 分组后的列表, 每个子列表的长度对应 group_sizes 中的值.

    示例:
        >>> elements = [1, 2, 3, 4, 5, 6]
        >>> group_sizes = [2, 1, 3]
        >>> group_elements(elements, group_sizes)
        [[1, 2], [3], [4, 5, 6]]
    """
    if len(elements) != sum(group_sizes):
        raise ValueError("元素总数必须等于 group_sizes 中数字的和.")
    
    result = []
    start_index = 0
    for size in group_sizes:
        # 切片得到当前组
        group = elements[start_index : start_index + size]
        result.append(group)
        start_index += size

    return result

def solution_is_valid(solution:str):
    pattern = (
        r'^'
        r'(?:.+\n<\|reserved_special_token_0\|>)*'
        r'.+\n'
        r'####\s'
        r'-?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\d+/\d+|\.\d+)'
        r'<\|reserved_special_token_0\|><\|eot_id\|>'
        r'$'
    )
    return re.fullmatch(pattern, solution) is not None

def get_eostep_indices(response_sequences: list[list[int]], step_split_token_id: int)->list[list[int]]:
    """
    获取每个响应序列中步骤分割符的索引.

    本函数旨在找出在每个响应序列中, 每个步骤结束（由特定的分割符标识）的位置索引.
    这对于处理分步骤的响应数据特别有用, 比如在分析对话系统中每个回复包含的步骤时.

    参数:
    response_sequences (list[list[int]]): 一个二维列表, 包含多个响应序列, 每个响应序列是由整数表示的token id序列.
    step_split_token_id (int): 用于标识每个步骤结束的特定token id.

    返回:
    list[list[int]]: 一个二维列表, 每个子列表包含对应响应序列中每个步骤结束的token id索引.
    """
    # 初始化用于存储所有响应序列中步骤分割符索引的列表
    eostep_indices = []

    # 遍历每个响应序列
    for response in response_sequences:
        # 初始化用于存储当前响应序列中步骤分割符索引的列表
        indices = []
        # 从响应序列的起始位置开始查找
        start = 0
        # 当前查找位置未超出响应序列长度时, 继续查找
        while start < len(response):
            try:
                # 查找 step_split_token_id 在 response 中的位置
                idx = response.index(step_split_token_id, start)
                # 将找到的位置索引添加到列表中
                indices.append(idx)
                # 更新下一次查找的起始位置
                start = idx + 1
            except ValueError:
                # 如果未找到 step_split_token_id, 则退出循环
                break
        # 将当前响应序列中所有步骤分割符的索引添加到结果列表中
        eostep_indices.append(indices)

    # 返回所有响应序列中步骤分割符的索引列表
    return eostep_indices

def preallocate_memory():
    # 获取当前GPU的显存总量
    total_mem = get_accelerator().total_memory()
    
    # 预留 10% 的显存作为安全余量，防止OOM
    reserve_mem = int(total_mem * 0.03)
    prealloc_size = total_mem - reserve_mem
    
    # 分配一个大的张量占用显存（保留引用防止被回收）
    dummy_tensor = torch.empty(
        (prealloc_size // 4,),  # 假设float32占4字节
        dtype=torch.float32,
        device=torch.cuda.current_device()
    )
    return dummy_tensor