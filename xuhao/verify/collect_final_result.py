import json

def accumulate_final_result():
    with open("xuhao/result.txt", 'r') as f:
        solutions = f.readlines()
    with open("xuhao/verification_result.json", 'r') as f:
        verifications = json.load(f)
    with open("xuhao/verification_label_plevel.json", 'r') as f:
        verification_labels = json.load(f)
    with open("xuhao/answer_label.json", 'r') as f:
        answer_labels = json.load(f)
    
    result = []

    for idx, solution in enumerate(solutions):
        solution = json.loads(solution)
        question = solution["question"]
        reference_answer = solution["reference answer"]
        answer = solution["answer"]
        verification_list = verifications[idx]
        answer_label = answer_labels[idx]
        verification_label = verification_labels[idx]
        result.append({
            "index": idx,
            "question": question,
            "reference answer": reference_answer,
            "answer": answer,
            "verification list": verification_list,
            "answer label": answer_label,
            "verification label": verification_label
        })
    
    with open("xuhao/final_result.json", 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    accumulate_final_result()