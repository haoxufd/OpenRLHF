import json
import re

solution_file = "/root/OpenRLHF/xuhao/solve/data/output/solution.json"
solution_label_file = "/root/OpenRLHF/xuhao/solve/data/output/solution_label.json"

def generate_solution_label():
    answer_label = []
    
    with open(solution_file, 'r') as f:
        solution_data = json.load(f)
    
    for item in solution_data:
        # reference answer may take the form as "10,800"
        ref_answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', item["reference_solution"])
        ref_answer[0] = ref_answer[0].replace(',', '')
        answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', item["solution"])

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
    
    with open(solution_label_file, 'w') as f:
        json.dump(answer_label, f, indent=4)


if __name__ == "__main__":
    generate_solution_label()