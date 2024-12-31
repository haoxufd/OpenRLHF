from email import message
import json
import os
import openai
from ray import get
from tqdm import tqdm

openai.api_key = "sk-DyZUsje31Q4rEeHfgo4BsA9IbEliawFNhzzt5SVaZ9rMN2Hu"
openai.base_url = "https://chat.cloudapi.vip/v1/"
openai.default_headers = {"x-foo": "true"}

def extract_concise_verification_result(num_problem=1e8):
    with open("xuhao/problem_desc.json", 'r') as f:
        problem_desc = json.load(f)

    with open("xuhao/verification_result.txt", 'r') as f:
        lines = f.readlines()

    result = []
    
    for idx, desc in enumerate(problem_desc):
        if idx >= num_problem:
            break
        num_steps = desc["num steps"]
        start_idx = desc["acc"]
        tmp = []
        for i in range(num_steps):
            line = lines[start_idx+i]
            if line.lower().startswith("\"evaluation: correct"):
                tmp.append(True)
            elif line.lower().startswith("\"evaluation: incorrect"):
                tmp.append(False)
            else:
                raise ValueError(f"Verification {idx} doesn't start with valid format")
        result.append(tmp)
    
    with open("xuhao/verification_result_concise.json", 'w') as f:
        json.dump(result, f, indent=4)

def generate_verification_label_plevel():
    result = []

    with open("xuhao/verification_result_concise.json", 'r') as f:
        verificaton_result = json.load(f)

    with open("xuhao/answer_label.json", 'r') as f:
        answer_label = json.load(f)
    
    for idx, label in enumerate(answer_label):
        result.append(label == all(verificaton_result[idx]))
    
    with open("xuhao/verification_label_plevel.json", 'w') as f:
        json.dump(result, f, indent=4)

def get_verification_accuracy_plevel():
    with open("xuhao/verification_label_plevel.json", 'r') as f:
        obj = json.load(f)

    cnt = 0
    for label in obj:
        if label == True:
            cnt += 1

    return cnt / len(obj)

def generate_verification_label_slevel(num_problems=1e8, api="claude"):
    with open("xuhao/answer_label.json", 'r') as f:
        answer_label = json.load(f)
    
    with open("xuhao/verification_result_concise.json", 'r') as f:
        verification_result = json.load(f)

    with open("xuhao/verification_dataset.json", 'r') as f:
        verification_data = json.load(f)
    
    with open("xuhao/verification_system_message.txt",'r') as f:
        system_message = f.read()
    
    with open("xuhao/verification_few_shot.json", 'r') as f:
        examples = json.load(f)
    
    assert len(answer_label) == len(verification_result)
    
    result = []
    claude_verification_result = []

    try:
        for idx, is_right in enumerate(tqdm(answer_label[:num_problems])):
            if idx >= num_problems:
                break
            cur_verification_result = []
            cur_claude_verification_result = []
            if is_right:
                # answer is right, verifications should all be right
                first_false = False
                for r in verification_result[idx]:
                    if first_false:
                        cur_verification_result.append(None)
                        continue
                    cur_verification_result.append(r)
                    if r == False:
                        first_false = True
            else:
                print(f"Answer {idx} is wrong, call claude to evaluate...")
                for data in verification_data[idx]["input"]:
                    messages = [{"role": "system", "content": system_message}]
                    
                    for example in examples:
                        messages.extend([
                            {
                                "role": "user",
                                "content": example["input"],
                            },
                            {
                                "role": "assistant",
                                "content": example["output"]
                            }
                        ])
                    
                    messages.append({
                        "role": "user",
                        "content": data
                    })

                    response = claude_generate(messages)
                    cur_claude_verification_result.append(response)
                    if "evaluation: incorrect" in response.lower():
                        break

                cur_claude_verification_result_bool = ["evaluation: correct" in x.lower() for x in cur_claude_verification_result]
                # compare cur_claude_verification_result_bool and verification_result[idx]
                for i, e in enumerate(cur_claude_verification_result_bool):
                    if e == verification_result[idx][i]:
                        cur_verification_result.append(True)
                    else:
                        cur_verification_result.append(False)
                        break
                
                while len(cur_verification_result) < len(verification_result[idx]):
                    cur_verification_result.append(None)
                
                while len(cur_claude_verification_result) < len(verification_result[idx]):
                    cur_claude_verification_result.append("")
            
            claude_verification_result.append(cur_claude_verification_result)
            result.append(cur_verification_result)
    except Exception as e:
        print(e)
    finally:
        with open("xuhao/verification_label_slevel.json", 'w') as f:
            json.dump(result, f, indent=4)
        
        with open("xuhao/claude_verification_result.json", 'w') as f:
            json.dump(claude_verification_result, f, indent=4)

def get_verification_accuracy_slevel():
    # If an answer is right, verification = [True, True, False, False]
    # total_cnt += 3, right_cnt += 2
    # If an answer is wrong, reference verification = [True, False], verification = [True, True, False]
    # total_cnt += 2, right_cnt += 1
    with open("xuhao/verification_label_slevel.json", 'r') as f:
        obj = json.load(f)
    
    total = 0
    right = 0
    for p in obj:
        for l in p:
            if l is None:
                break
            if l == True:
                right += 1
            total += 1
    
    return right / total
    

def claude_generate(messages):
    completion = openai.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages
    )
    return completion.choices[0].message.content.strip()

def postprocess_verification_result(num_problem=1e8):
    with open("xuhao/verification_result.txt", 'r') as f:
        lines = f.readlines()

    with open("xuhao/problem_desc.json", 'r') as f:
        problem_desc = json.load(f)

    result = []
    for idx, pd in enumerate(problem_desc):
        if idx >= num_problem:
            break
        result.append(lines[pd["acc"]: pd["acc"] + pd["num steps"]])

    with open("xuhao/verification_result.json", 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    # generate_verification_label_plevel()
    print(get_verification_accuracy_plevel())
