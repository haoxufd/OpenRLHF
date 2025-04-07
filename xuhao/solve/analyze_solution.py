from ast import arg
import glob
import json
import re
from xuhao.utils import get_solve_result
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

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

def visualize_acc(csv_path, output_name="accuracy_plot.png"):

    df = pd.read_csv(csv_path)
    if "Acc" not in df.columns:
        raise ValueError("row 'Acc' not in the csv file")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(df)+1), 
            df["Acc"],
            marker='o',
            color='#E74C3C',
            linewidth=2)
    
    ax.set_title("Model Accuracy Progression", fontsize=14, pad=15)
    ax.set_xlabel("Eval Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')  # 关键修改点
    plt.close(fig)  
def calc_ppo_eval_result(args):
    ref_solutions = load_dataset("openai/gsm8k", "main")["test"]["answer"][:args.num_eval_data]
    # 获取 args.eval_output_dir 下的所有名为 eval_step_i.json 的文件
    solution_files = glob.glob(os.path.join(args.eval_output_dir, "eval_step_*.json"))
    solution_files.sort(key=lambda x: int(re.search(r'eval_step_(\d+)\.json$', x).group(1)))
    
    _calc_ppo_eval_result(solution_files, ref_solutions, args.eval_output_dir + "/result.csv", args.print_incorrect_indices)

    csv_path = os.path.join(args.eval_output_dir, "result.csv")

    # 新增可视化调用
    plot_path = os.path.join(args.eval_output_dir, "accuracy_curve.png")
    visualize_acc(csv_path, plot_path)  # 调用可视化函数
    print(f"Visualization saved to {plot_path}")

def calc_single_solve_acc(solution_file):
    ref_solutions = load_dataset("openai/gsm8k", "main")["test"]["answer"]
    with open(solution_file, "r") as f:
        solutions = json.load(f)
    result = get_solve_result(solutions, ref_solutions)

    print(f"Total: {result[2]}")
    print(f"Num Correct: {result[0]}")
    print(f"Num Incorrect: {result[1]}")
    print(f"Accuracy: {result[3]}")
    print(f"Incorrect Solutions: {result[4]}") 

if __name__ == "__main__":
    # python xuhao/solve/analyze_solution.py --num_eval_data=100 --num_eval=6 --eval_output_dir=/root/data/ppo/eval_output_1
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument("--num_eval_data", type=int, default=100)
    paser.add_argument("--num_eval", type=int, default=10)
    paser.add_argument("--eval_output_dir", type=str, default="/root/data/ppo/eval_output_1")
    paser.add_argument("--print_incorrect_indices", action="store_true", default=False)
    
    paser.add_argument("--solution_file", type=str, default=None)

    args = paser.parse_args()

    if args.solution_file is not None:
        calc_single_solve_acc(args.solution_file)
    else:
        calc_ppo_eval_result(args)
