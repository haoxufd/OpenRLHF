import json
import re
from xuhao.utils import get_solve_result
import csv

from datasets import load_dataset

home_dir = "/home/user"


def generate_solution_label(
        problem_file=f"{home_dir}/OpenRLHF/xuhao/solve/data/input/gsm8k.json",
        solution_file = f"{home_dir}/OpenRLHF/xuhao/solve/data/output/solution.json",
        solution_label_file = f"{home_dir}/OpenRLHF/xuhao/solve/data/output/solution_label.json"):
    solution_label = []
    
    with open(solution_file, 'r') as f1, open(problem_file, 'r') as f2:
        solution_data = json.load(f1)
        problem_data = json.load(f2)
    
    for item in solution_data:
        # reference answer may take the form as "10,800"
        reference_solution = problem_data[item["problem_index"]]["reference_solution"]
        ref_answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', reference_solution)
        ref_answer[0] = ref_answer[0].replace(',', '')
        answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', item["solution"])

        assert len(ref_answer) == 1
        assert len(answer) < 2
        if len(answer) == 0:
            # cannot get an result value, ends with last step, not "#### xxx"
            # this kind of answer is definitely wrong, since reference answer 
            # can always produce a final value
            solution_label.append({
                "problem_index": item["problem_index"],
                "solution_label": False
            })
        else:
            # length = 1
            answer[0] = answer[0].replace(',', '')
            solution_label.append({
                "problem_index": item["problem_index"],
                "solution_label": float(ref_answer[0]) == float(answer[0])
            })
    
    with open(solution_label_file, 'w') as f:
        json.dump(solution_label, f, indent=4)

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
    ref_solutions = load_dataset("openai/gsm8k", "main")["test"]["answer"]
    solutions_files = [f"/root/data/eval_output/eval_step_{i}.json" for i in range(7)]
    calc_ppo_eval_result(solutions_files, ref_solutions, "/root/data/eval_output/result.csv")