import json
import re

solution_dataset_file = "xuhao/solution_dataset.json"
verification_dataset_file = "xuhao/verification_dataset.json"
answer_label_file = "xuhao/answer_label.json"

def generate_verification_dataset():
    with open(solution_dataset_file, 'r') as f:
        solution_data = json.load(f)

    data = []
    for item in solution_data:
        ref_answer = re.findall(r'#### ([+-]?\d*\.?\d+)', item["reference answer"])
        answer = re.findall(r'#### ([+-]?\d*\.?\d+)', item["answer"])

        assert len(ref_answer) == 1
        assert len(answer) < 2
        if len(answer) == 0:
            # cannot get an result value, ends with last step, not "#### xxx"
            # this kind of answer is definitely wrong, since reference answer 
            # can always produce a final value
            steps = item["answer"].split('\n')
        else:
            # length = 1
            steps = item["answer"].split('\n')[:-1]

        for i in range(len(steps)):
            verification_input = "Problem:\n{}\n\nReference Solution:\n{}\n\nPrevious Steps:\n{}\n\nStep to Evaluate:\n{}".format(item["question"], item["reference answer"], '\n'.join(steps[:i]), steps[i])
            data.append({
                "problem index": item["index"],
                "step index": i,
                "verification_input": verification_input
            })
    
    with open(verification_dataset_file, 'w') as f:
        json.dump(data, f, indent=4)

def generate_answer_label():
    answer_label = []
    
    with open(solution_dataset_file, 'r') as f:
        solution_data = json.load(f)
    
    for item in solution_data:
        # reference answer may take the form as "10,800"
        ref_answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', item["reference answer"])
        ref_answer[0] = ref_answer[0].replace(',', '')
        answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', item["answer"])

        assert len(ref_answer) == 1
        assert len(answer) < 2
        if len(answer) == 0:
            # cannot get an result value, ends with last step, not "#### xxx"
            # this kind of answer is definitely wrong, since reference answer 
            # can always produce a final value
            answer_label.append(False)
        else:
            # length = 1
            answer[0] = answer[0].replace(',', '')
            answer_label.append(float(ref_answer[0]) == float(answer[0]))
    
    with open(answer_label_file, 'w', encoding="utf-8") as f:
        json.dump(answer_label, f, indent=4)

if __name__ == "__main__":
    generate_verification_dataset()