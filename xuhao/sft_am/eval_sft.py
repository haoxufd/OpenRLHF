import json
import re
from xuhao.utils import read_json_list_file, write_json_list_file
from datasets import load_dataset
import csv
from xuhao.utils import get_solve_result

def get_acc_change(solution_file_list: list, ref_solution: list):
    """
    
    """
    change = []
    for file in solution_file_list:
        solution = read_json_list_file(file)
        change.append(get_solve_result(solution, ref_solution))
    return change

def main():
    solution_files = []
    for i in range(2, 20):
        solution_files.append(f"xuhao/sft_am/data/output/eval_step_{i}.json")
    data = load_dataset("openai/gsm8k", "main")
    ref_solution = data["test"]["answer"]

    acc_change = get_acc_change(solution_files, ref_solution)
    acc_change.insert(0, ["Num Correct", "Num Incorrect"])
    with open("xuhao/sft_am/data/output/acc_change.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(acc_change)

if __name__ == "__main__":
    solution = read_json_list_file("xuhao/solve/data/output/solution_llama-instruct_gsm8k-test.json")
    data = load_dataset("openai/gsm8k", "main")
    ref_solution = data["test"]["answer"]
    result = get_solve_result(solution, ref_solution)
    print(result)
    print(result[0] / sum(result))