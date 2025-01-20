import json
from itertools import groupby
from unittest import result
from datasets import interleave_datasets, load_dataset
import re

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

def get_solve_result(solution: list, ref_solution: list):
    """
    由于是通过 batch inference 得到的 solution, 其中的数据可能多于 ref_solution
    但两者数据是按顺序对应的, 遍历的时候以 ref_solution 为主即可
    """
    num_correct = num_incorrect = 0
    for i in range(len(ref_solution)):
        answer = solution[i]
        ref_answer = ref_solution[i]
        
        value = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', answer)
        value = value[0].replace(',', '') if len(value) > 0 else None
        ref_value = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', ref_answer)
        assert len(ref_value) > 0
        ref_value = ref_value[0].replace(',', '')

        if value is None:
            num_incorrect += 1
        elif value is not None and float(value) != float(ref_value):
            num_incorrect += 1
        else:
            num_correct += 1
    return [num_correct, num_incorrect]

def get_steps(solution: str):
    """
    """
    steps = solution.strip().split('\n')
    result = []
    for step in steps:
        if step:
            result.append(step)
    return result[:-1] if "####" in result[-1] else result

def get_final_value_from_solution(solution: str) -> float | None:
    content = solution.split("####")[-1] if "####" in solution else solution.split('\n')[-1]
    content = content.strip()
    numbers = find_numbers(content)
    return None if not numbers else numbers[0]

def find_newline_indices(s):
    indices = []
    for i, char in enumerate(s):
        if char == '\n':
            # 如果是第一个 '\n' 或与前一个字符不连续
            if i == 0 or s[i - 1] != '\n':
                indices.append(i)
    return indices

def find_numbers(text: str):
    pattern = r'-?(?:\d*\.?\d+|\.\d+)(?:,\d{3})*'
    res = re.findall(pattern, text)
    res = [float(x.replace(',', '')) for x in res]
    return res

def group_elements(elements, group_sizes):
    """
    将列表 elements 中的元素按照 group_sizes 中指定的分组大小分组。

    参数:
        elements (list): 待分组的元素列表，其长度必须等于 sum(group_sizes)。
        group_sizes (list of int): 每组需要的元素个数列表，总和应与 elements 的长度相等。

    返回:
        list of list: 分组后的列表，每个子列表的长度对应 group_sizes 中的值。

    示例:
        >>> elements = [1, 2, 3, 4, 5, 6]
        >>> group_sizes = [2, 1, 3]
        >>> group_elements(elements, group_sizes)
        [[1, 2], [3], [4, 5, 6]]
    """
    if len(elements) != sum(group_sizes):
        raise ValueError("元素总数必须等于 group_sizes 中数字的和。")
    
    result = []
    start_index = 0
    for size in group_sizes:
        # 切片得到当前组
        group = elements[start_index : start_index + size]
        result.append(group)
        start_index += size

    return result

def solution_end_is_valid(solution: str):
    steps = solution.split('\n')
    last_step = steps[-1].strip()
    if not (re.match(r"^####\s*-?[\d,]*(\.[\d,]+)?$", last_step) or re.match(r"^####\s+None$", last_step)):
        return False
    
    return True