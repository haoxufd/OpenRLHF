import json
from itertools import groupby
import copy
from xuhao.utils import read_json_list_file, write_json_list_file
from xuhao.utils import get_grouped_data

def extract_concise_verification_result(verification_result_file = "xuhao/verify/data/output/verification_result_claude.json"):
    with open(verification_result_file, 'r') as f:
        verification_result = json.load(f)
    
    grouped_result = [list(group) for key, group in groupby(verification_result, key=lambda x: x["problem_index"])]

    concise_grouped_result = []
    for item in grouped_result:
        tmp = []
        for data in item:
            tmp.append("INCORRECT" not in data["verification_result"])
        concise_grouped_result.append(tmp)
    
    return concise_grouped_result

def get_verification_result_plevel(
        verification_result_file = "xuhao/verify/data/output/verification_result_claude.json",
        solution_label_file = "xuhao/solve/data/output/solution_label.json",
        verification_result_label_file = None):
    concise_verification_result = extract_concise_verification_result(verification_result_file)
    solution_label = read_json_list_file(solution_label_file)
    
    result = []
    num_right_verification = 0
    cc = ci = ic = ii = 0
    for idx, label in enumerate(solution_label):
        verification_result_label = label["solution_label"] == all(concise_verification_result[idx])
        if verification_result_label:
            num_right_verification += 1
        if label["solution_label"] and verification_result_label:
            cc += 1
        elif label["solution_label"] and not verification_result_label:
            ci += 1
        elif not label["solution_label"] and verification_result_label:
            ii += 1
        else:
            ic += 1
        result.append({
            "problem_index": idx,
            "verification_result_label": verification_result_label
        })
    
    assert len(solution_label) == (cc + ci + ic + ii)
    
    if verification_result_label_file is not None:
        write_json_list_file(verification_result_label_file, result)
        print("Verification result label has been written to ", verification_result_label_file)
    
    return [cc, ci, ic, ii]

def get_verification_result_slevel(
    verification_result_file,
    ref_verification_result_file):
    """
    verification_result_file 是部分 verification
    ref_verification_result_file 是全部的 verification, 是从 gpt 蒸馏得来的
    虽然 ref_verification_result_file 中也有错误数据, 但若 verification_result_file 
    对应的是 sft data, 那么 ref_verification_result_file 中与 verification_result_file 
    对应的就都是正确的, 因为 sft data 选取的都是蒸馏而来的正确的 verification result
    """
    result = read_json_list_file(verification_result_file)
    ref_result = read_json_list_file(ref_verification_result_file)
    ref_result = get_grouped_data(ref_result)

    num_right_slevel_verification = 0
    cc = ci = ic = ii = 0
    for data in result:
        v1 = "Evaluation Result: CORRECT" in ref_result[data["problem_index"]][data["step_index"]]["verification_result"]
        v2 = "Evaluation Result: CORRECT" in data["verification_result"]
        if v1 == v2:
            num_right_slevel_verification += 1
        if v1 and v2:
            cc += 1
        elif v1 and not v2:
            ci += 1
        elif not v1 and v2:
            ic += 1
        else:
            ii += 1
    
    # print("Num SLevel Verifications: ", len(result))
    # print("Num Right SLevel Verifications: ", num_right_slevel_verification)
    # print("SLevel ACC: ", num_right_slevel_verification / len(result))
    # print("Verify CORRECT Step to be CORRECT: ", cc)
    # print("Verify CORRECT Step to be INCORRECT: ", ci)
    # print("Verify INCORRECT Step to be CORRECT: ", ic)
    # print("Verify INCORRECT Step to be INCORRECT: ", ii)

    return [cc, ci, ic, ii]

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
    # get_verification_result_plevel(
    #     verification_result_file="xuhao/verify/data/output/verification_result_claude_new.json",
    #     solution_label_file="xuhao/solve/data/output/solution_label_new.json",
    #     verification_result_label_file="xuhao/verify/data/output/verification_result_label_claude_new.json")
    
    result = get_verification_result_plevel("xuhao/verify/data/output/verification_result_sft_test_new_ckpt2.json", "xuhao/sft_prm/data/input/solution_label_test_new.json")
    total = sum(result)
    right = result[0] + result[3]
    print("Problem Level:")
    print(result)
    print("Total: ", total, "Right: ", right, "Acc: ", right / total)
    print("Positive Recall: ", result[0] / (result[0] + result[1]))
    print("Negative Recall: ", result[3] / (result[2] + result[3]))
    print("Positive Precision: ", result[0] / (result[0] + result[2]))
    print("Negative Precision: ", result[3] / (result[1] + result[3]))

    result = get_verification_result_slevel("xuhao/verify/data/output/verification_result_sft_test_new_ckpt2.json", "xuhao/verify/data/output/verification_result_claude_new.json")
    total = sum(result)
    right = result[0] + result[3]
    print("Step Level:")
    print(result)
    print("Total: ", total, "Right: ", right, "Acc: ", right / total)
    print("Positive Recall: ", result[0] / (result[0] + result[1]))
    print("Negative Recall: ", result[3] / (result[2] + result[3]))
    print("Positive Precision: ", result[0] / (result[0] + result[2]))
    print("Negative Precision: ", result[3] / (result[1] + result[3]))