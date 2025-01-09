import json
from itertools import groupby

verification_result_file = "xuhao/verify/data/output/verification_result_claude.json"
verification_result_label_file = "xuhao/verify/data/output/verification_result_label_claude.json"
solution_label_file = "xuhao/solve/data/output/solution_label.json"

def extract_concise_verification_result():
    with open(verification_result_file, 'r') as f:
        verification_result = json.load(f)
    
    grouped_result = [list(group) for key, group in groupby(verification_result, key=lambda x: x["problem_index"])]

    concise_grouped_result = []
    for item in grouped_result:
        tmp = []
        for data in item:
            assert data["verification_result"].endswith("Evaluation Result: CORRECT") or data["verification_result"].endswith("Evaluation Result: INCORRECT")
            tmp.append(data["verification_result"].endswith("Evaluation Result: CORRECT"))
        concise_grouped_result.append(tmp)
    
    return concise_grouped_result

def get_verification_accuracy():
    concise_verification_result = extract_concise_verification_result()

    with open(solution_label_file, 'r') as f:
        solution_label = json.load(f)
    
    result = []
    num_right_verification = 0
    for idx, label in enumerate(solution_label):
        verification_result_label = label == all(concise_verification_result[idx])
        if verification_result_label:
            num_right_verification += 1
        result.append({
            "problem_index": idx,
            "verification_result_label": verification_result_label
        })
    
    print("Problem Num: ", len(solution_label))
    print("Num Right Verification: ", num_right_verification)
    print("Verification Accuracy: ", num_right_verification / len(solution_label))
    
    with open(verification_result_label_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    print("Verification result label has been written to ", verification_result_label_file)

if __name__ == "__main__":
    get_verification_accuracy()
