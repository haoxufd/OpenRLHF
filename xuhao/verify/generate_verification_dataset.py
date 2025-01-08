import json
import re

solution_file = "/root/OpenRLHF/xuhao/solve/data/output/solution_test.json"
verification_data_file = "/root/OpenRLHF/xuhao/verify/data/input/verification_data_test.json"

def generate_verification_dataset():
    with open(solution_file, 'r') as f:
        solution_data = json.load(f)

    data = []
    for item in solution_data:
        ref_answer = re.findall(r'#### ([+-]?\d*\.?\d+)', item["reference_solution"])
        answer = re.findall(r'#### ([+-]?\d*\.?\d+)', item["solution"])

        assert len(ref_answer) == 1
        assert len(answer) < 2
        if len(answer) == 0:
            # cannot get an result value, ends with last step, not "#### xxx"
            # this kind of answer is definitely wrong, since reference answer 
            # can always produce a final value
            steps = item["solution"].split('\n')
        else:
            # length = 1
            steps = item["solution"].split('\n')[:-1]

        for i in range(len(steps)):
            verification_input = "Problem:\n{}\n\nReference Solution:\n{}\n\nPrevious Steps:\n{}\n\nStep to Evaluate:\n{}".format(item["problem"], item["reference_solution"], '\n'.join(steps[:i]), steps[i])
            data.append({
                "problem_index": item["index"],
                "step_index": i,
                "verification_input": verification_input
            })
    
    with open(verification_data_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    pass