import json
import re
from xuhao.utils import get_solve_result
import csv

from datasets import load_dataset

home_dir = "/home/user"

def calc_ppo_eval_result(solution_files: list[str], ref_solutions: list[str], output_file: str):
    """
    统计 ppo 训练解题模型过程中模型解题正确率的变化
    """
    results = []
    for solution_file in solution_files:
        with open(solution_file, 'r') as f:
            solutions = json.load(f)
        results.append(get_solve_result(solutions, ref_solutions))
    
    results.insert(0, ["Num Correct", "Num Incorrect", "Total", "Acc", "Incorrect Indices"])
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        for result in results:
            row = [str(item) if isinstance(item, list) else item for item in result]
            writer.writerow(row)
    
    print(f"Eval result has been successfully writen to {output_file}")
    

if __name__ == "__main__":
    # ref_solutions = load_dataset("openai/gsm8k", "main")["test"]["answer"]
    # solutions_files = [f"/root/data/eval_output/eval_step_{i}.json" for i in range(7)]
    # calc_ppo_eval_result(solutions_files, ref_solutions, "/root/data/eval_output/result.csv")

    # ref_solutions = load_dataset("openai/gsm8k", "main")["test"]["answer"]
    # with open("xuhao/solve/data/output/solution_llama-instruct-special-token_gsm8k-test.json", 'r') as f:
    #     solutions = json.load(f)
    # result = get_solve_result(solutions, ref_solutions)
    # print(result[:4])

