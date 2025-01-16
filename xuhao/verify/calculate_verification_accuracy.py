import json
from itertools import groupby
import copy

def extract_concise_verification_result(verification_result_file = "xuhao/verify/data/output/verification_result_claude.json"):
    with open(verification_result_file, 'r') as f:
        verification_result = json.load(f)
    
    grouped_result = [list(group) for key, group in groupby(verification_result, key=lambda x: x["problem_index"])]

    concise_grouped_result = []
    for item in grouped_result:
        tmp = []
        for data in item:
            # assert ("Evaluation Result: CORRECT" in data["verification_result"]) or ("Evaluation Result: INCORRECT" in data["verification_result"])
            tmp.append("INCORRECT" not in data["verification_result"])
        concise_grouped_result.append(tmp)
    
    return concise_grouped_result

def get_verification_accuracy_plevel(
        verification_result_file = "xuhao/verify/data/output/verification_result_claude.json",
        solution_label_file = "xuhao/solve/data/output/solution_label.json",
        verification_result_label_file = None):
    concise_verification_result = extract_concise_verification_result(verification_result_file)

    with open(solution_label_file, 'r') as f:
        solution_label = json.load(f)
    
    result = []
    num_right_verification = 0
    cc = 0
    ci = 0
    ic = 0
    ii = 0
    for idx, label in enumerate(solution_label):
        verification_result_label = label == all(concise_verification_result[idx])
        if verification_result_label:
            num_right_verification += 1
        if label and verification_result_label:
            cc += 1
        elif label and not verification_result_label:
            ci += 1
        elif not label and verification_result_label:
            ic += 1
        else:
            ii += 1
        result.append({
            "problem_index": idx,
            "verification_result_label": verification_result_label
        })
    
    print("Problem Num: ", len(solution_label))
    print("Num Right Verification: ", num_right_verification)
    print("Verification Accuracy: ", num_right_verification / len(solution_label))
    print(f"Verify CORRECT Solution to be CORRECT: {cc}")
    print(f"Verify CORRECT Solution to be INCORRECT: {ci}")
    print(f"Verify INCORRECT Solution to be CORRECT: {ii}")
    print(f"Verify INCORRECT Solution to be INCORRECT: {ic}")
    
    if verification_result_label_file is not None:
        with open(verification_result_label_file, 'w') as f:
            json.dump(result, f, indent=4)
        print("Verification result label has been written to ", verification_result_label_file)

def get_verification_accuracy_slevel(
        verification_result_file = "xuhao/verify/data/output/verification_result_claude.json"):
    with open(verification_result_file, 'r') as f:
        result = json.load(f)

    num_right_slevel_verification = 0
    cc = ci = ic = ii = 0
    for data in result:
        v1 = "Evaluation Result: CORRECT" in data["output"]
        v2 = "Evaluation Result: CORRECT" in data["verification_result"]
        if v1 == v2:
            num_right_slevel_verification += 1
        if v1 and v2:
            cc += 1
        elif v1 and not v2:
            ci += 2
        elif not v1 and v2:
            ic += 1
        else:
            ii += 1
    
    print("Num SLevel Verifications: ", len(result))
    print("Num Right SLevel Verifications: ", num_right_slevel_verification)
    print("SLevel ACC: ", num_right_slevel_verification / len(result))
    print("Verify CORRECT Step to be CORRECT: ", cc)
    print("Verify CORRECT Step to be INCORRECT: ", ci)
    print("Verify INCORRECT Step to be CORRECT: ", ic)
    print("Verify INCORRECT Step to be INCORRECT: ", ii)

def postprocess_eval_result():
    with open("xuhao/sft/data/input/sft_data.json", 'r') as f:
        sft_data = json.load(f)
    sft_test_data = sft_data["test"]

    for i in range(20):
        with open(f"xuhao/sft/data/output/eval_step_{i}.json", 'r') as f:
            eval_result = json.load(f)
        
        verification_result = []
        for idx, data in enumerate(sft_test_data):
            new_data = copy.deepcopy(data)
            new_data["verification_result"] = eval_result[idx]
            verification_result.append(new_data)
        
        with open(f"xuhao/sft/data/output/verification_result_llama-3.1-8b-sft_{i}.json", 'w') as f:
            json.dump(verification_result, f, indent=4)

if __name__ == "__main__":
    get_verification_accuracy_plevel(
        verification_result_file="xuhao/verify/data/output/verification_result_claude_new.json",
        solution_label_file="xuhao/solve/data/output/solution_label_new.json",
        verification_result_label_file="xuhao/verify/data/output/verification_result_label_claude_new.json")