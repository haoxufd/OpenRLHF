import json

verification_result_file = "xuhao/llama-3.1-8b-sft_verification_result_eval.json"
concise_verification_result_file = verification_result_file.replace(".json", "_concise.json")
verification_label_file = "xuhao/llama-3.1-8b-sft_verification_label_eval.json"

eval_start_problem = 7162

def extract_concise_verification_result(num_problem=1e8):
    with open(verification_result_file, 'r') as f:
        verification_result = json.load(f)

    concise_result = [[x.lower().strip().startswith("evaluation: correct") for x in item] for item in verification_result]
    
    with open(concise_verification_result_file, 'w') as f:
        json.dump(concise_result, f, indent=4)

def generate_verification_label_plevel(start_idx):
    result = []

    with open(concise_verification_result_file, 'r') as f:
        verificaton_result = json.load(f)

    with open("xuhao/answer_label.json", 'r') as f:
        answer_label = json.load(f)[start_idx:]
    
    for idx, label in enumerate(answer_label):
        result.append(label == all(verificaton_result[idx]))
    
    with open(verification_label_file, 'w') as f:
        json.dump(result, f, indent=4)

def get_verification_accuracy_plevel():
    with open(verification_label_file, 'r') as f:
        obj = json.load(f)

    cnt = 0
    for label in obj:
        if label == True:
            cnt += 1

    return cnt / len(obj)

def postprocess_verification_result(start_idx):
    with open("xuhao/llama-3.1-8b-sft_verification_result_eval.txt", 'r') as f:
        raw_verification_result = f.readlines()
    
    tmp = []
    for line in raw_verification_result:
        tmp.append(line.strip('\"\n '))
    
    with open("xuhao/problem_desc.json", 'r') as f:
        problem_desc = json.load(f)
    
    result = []

    acc = 0
    for desc in problem_desc[start_idx:]:
        result.append(tmp[acc:acc+desc["num steps"]])
        acc += desc["num steps"]
    
    with open(verification_result_file, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    extract_concise_verification_result()
    generate_verification_label_plevel(eval_start_problem)
    print("Acc: ", get_verification_accuracy_plevel())
