import json
from itertools import groupby
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
    acc = num_correct / total
    return [num_correct, num_incorrect, total, acc, incorrect_indices]