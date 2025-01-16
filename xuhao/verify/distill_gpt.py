import json
from openai import OpenAI

client = OpenAI(
    base_url='https://api.feidaapi.com/v1',
    api_key="sk-sOawzVByUJ8rjGrkgNTlMOPde1m1i40A3ty80EhtBeETn2Ui"
)

def print_item_num(json_file):
    with open(json_file, 'r') as f:
        print(len(json.load(f)))

def read_json_list_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def write_json_list_file(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def gpt_generate(messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=messages)
    return completion.choices[0].message.content.strip()

def get_redistill_problem(solution_file, old_solution_file):
    """
    获取需要重蒸馏的数据, solution 中跟 old_solution 重复的不需要重新蒸馏, 反正需要重新蒸馏
    返回一个 hash table, 长度为 problem 总数, 置为 True 的是需要重新蒸馏的
    """
    solution = read_json_list_file(solution_file)
    old_solution = read_json_list_file(old_solution_file)
    assert len(solution) == len(old_solution)

    num = len(solution)
    ret = [False] * num
    cnt = 0
    for i in range(num):
        if solution[i]["solution"] != old_solution[i]["solution"]:
            ret[i] = True
            cnt += 1
    print(cnt)
    return ret

def generate_verification_result_gpt(
        verification_data_file="xuhao/verify/data/input/verification_data_new.json",
        verification_result_file="xuhao/verify/data/output/verification_result_claude.json",
        verification_system_message_file="xuhao/verify/data/input/verification_system_message.txt",
        verification_few_shot_file="xuhao/verify/data/input/verification_few_shot.json",
        solution_file="xuhao/solve/data/output/solution_new.json",
        old_solution_file="xuhao/solve/data/output/solution_old.json",
        num_verifications=1e8, 
        num_few_shots=1, 
        save_batch=20):
    with open(verification_data_file, 'r') as f1, open(verification_result_file, 'r') as f2:
        verification_data = json.load(f1)
        verification_result = json.load(f2)

    with open(verification_system_message_file, 'r') as f1, open(verification_few_shot_file, 'r') as f2:
        verification_system_message = f1.read()
        examples = json.load(f2)

    messages = [{"role": "system", "content": verification_system_message}]           
    for example in examples[:num_few_shots]:
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
    
    # 找到开始蒸馏的 problem step
    cnt = 0
    tag = get_redistill_problem(solution_file, old_solution_file)
    for idx, data in enumerate(verification_data):
        if cnt == len(verification_result):
            start_idx = idx
            break
        if tag[data["problem_index"]]:
            cnt += 1

    end_idx = min(start_idx + num_verifications, len(verification_data))

    cnt = 0
    for idx, data in enumerate(verification_data[start_idx:end_idx]):
        if not tag[data["problem_index"]]:
            continue
        print("Processing problem ", data["problem_index"], " step", data["step_index"])
        if cnt % save_batch == 0 and cnt != 0:
            with open(verification_result_file, 'w') as f:
                json.dump(verification_result, f, indent=4)

        if len(verification_result) > 0:
            last_response = verification_result[-1]["verification_result"]
            skip = data["problem_index"] == verification_result[-1]["problem_index"] and (
                last_response == "NIL" or last_response.lower().strip().endswith("incorrect")
            )
        else:
            skip = False

        if skip:
            response = "NIL"
        else:
            messages.append({
                "role": "user",
                "content": data["verification_input"]
            })
            response = gpt_generate(messages=messages)
            messages.pop()
            cnt += 1
        
        verification_result.append({
                "problem_index": data["problem_index"],
                "step_index": data["step_index"],
                "verification_result": response
        })
    
    with open(verification_result_file, 'w') as f:
        json.dump(verification_result, f, indent=4)

def postprocess_verification_result(
        verification_result_file="xuhao/verify/data/output/verification_result_claude.json"):
    verification_result = read_json_list_file(verification_result_file)

    result = []
    for data in verification_result:
        if data["verification_result"] != "NIL":
            result.append(data)
    
    write_json_list_file(verification_result_file, result)

if __name__ == "__main__":
    generate_verification_result_gpt()
