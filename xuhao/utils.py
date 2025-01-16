import json
from itertools import groupby

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