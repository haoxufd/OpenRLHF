import json
import re
from xuhao.utils import get_grouped_data

home_dir = "/home/user"

def print_item_num(json_file):
    with open(json_file, 'r') as f:
        print(len(json.load(f)))

def read_json_list_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def write_json_list_file(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def get_steps_from_solution(solution: str):
    assert "####" in solution
    assert solution.startswith("1. ")

    points = [0]
    step_idx = 2
    while True:
        point = solution.find(f"\n{step_idx}. ")
        if point < 0:
            break
        points.append(point + 1)
        step_idx += 1
    points.append(solution.find("####"))

    steps = []
    for i in range(len(points) - 1):
        steps.append(solution[points[i] : points[i + 1]].strip())
    
    return steps

def generate_verification_dataset(
        problem_file=f"{home_dir}/OpenRLHF/xuhao/solve/data/input/gsm8k.json",
        solution_file=f"{home_dir}/OpenRLHF/xuhao/solve/data/output/solution.json",
        verification_data_file=f"{home_dir}/OpenRLHF/xuhao/verify/data/input/verification_data.json"):
    problem = read_json_list_file(problem_file)
    solution = read_json_list_file(solution_file)

    data = []
    for idx, item in enumerate(solution):
        steps = get_steps_from_solution(item["solution"])

        for i in range(len(steps)):
            verification_input = "Problem:\n{}\n\nReference Solution:\n{}\n\nPrevious Steps:\n{}\n\nStep to Evaluate:\n{}".format(problem[idx]["problem"], problem[idx]["reference_solution"], '\n'.join(steps[:i]), steps[i])
            data.append({
                "problem_index": item["problem_index"],
                "step_index": i,
                "verification_input": verification_input
            })
    
    with open(verification_data_file, 'w') as f:
        json.dump(data, f, indent=4)

def merge_verification_result(verification_result_file_1, verification_result_file_2, verification_result_file):
    """
    把 file 1 中的替换成 file 2 中的
    """
    verification_result_1 = read_json_list_file(verification_result_file_1)
    verification_result_2 = read_json_list_file(verification_result_file_2)
    
    grouped_result_1 = get_grouped_data(verification_result_1)
    grouped_result_2 = get_grouped_data(verification_result_2)
    
    use_result_1 = [True] * len(grouped_result_1)
    for data in grouped_result_2:
        use_result_1[data[0]["problem_index"]] = False
    
    grouped_result = []
    for data in grouped_result_1:
        if use_result_1[data[0]["problem_index"]]:
            grouped_result.append(data)
    grouped_result += grouped_result_2

    grouped_result.sort(key=lambda x: x[0]["problem_index"])

    result = []
    for x in grouped_result:
        for y in x:
            result.append(y)
    
    write_json_list_file(verification_result_file, result)
    
if __name__ == "__main__":
    merge_verification_result(
        "xuhao/verify/data/output/verification_result_claude_old.json",
        "xuhao/verify/data/output/verification_result_claude.json",
        "xuhao/verify/data/output/verification_result_claude_new.json")