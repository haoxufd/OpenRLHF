from ast import arg
import glob
import json
import re
from xuhao.utils import get_solve_result
import csv
import os

from datasets import load_dataset

def _calc_ppo_eval_result(
        solution_files: list[str], 
        ref_solutions: list[str], 
        output_file: str,
        print_incorrect_indices=False):
    """
    统计 ppo 训练解题模型过程中模型解题正确率的变化
    """
    results = []
    for solution_file in solution_files:
        with open(solution_file, 'r') as f:
            solutions = json.load(f)
        results.append(get_solve_result(solutions, ref_solutions))
    
    results.insert(0, ["Num Correct", "Num Incorrect", "Total", "Acc", "Incorrect Indices"])
    if not print_incorrect_indices:
        results = [x[: -1] for x in results]
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        for result in results:
            row = [str(item) if isinstance(item, list) else item for item in result]
            writer.writerow(row)
    
    print(f"Eval result has been successfully writen to {output_file}")

def calc_ppo_eval_result(args):
    ref_solutions = load_dataset("openai/gsm8k", "main")["test"]["answer"][:args.num_eval_data]
    # 获取 args.eval_output_dir 下的所有名为 eval_step_i.json 的文件
    solution_files = glob.glob(os.path.join(args.eval_output_dir, "eval_step_*.json"))
    solution_files.sort(key=lambda x: int(re.search(r'eval_step_(\d+)\.json$', x).group(1)))
    
    _calc_ppo_eval_result(solution_files, ref_solutions, args.eval_output_dir + "/result.csv", args.print_incorrect_indices)

if __name__ == "__main__":
    # python xuhao/solve/analyze_solution.py --num_eval_data=100 --num_eval=6 --eval_output_dir=/root/data/ppo/eval_output_1
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument("--num_eval_data", type=int, default=100)
    paser.add_argument("--num_eval", type=int, default=10)
    paser.add_argument("--eval_output_dir", type=str, default="/root/data/ppo/eval_output_1")
    paser.add_argument("--print_incorrect_indices", action="store_true", default=False)

    calc_ppo_eval_result(paser.parse_args())
