import json
from openai import OpenAI

client = OpenAI(
    base_url='https://api.feidaapi.com/v1',
    api_key="sk-YC4GnWLQ8dI6qDqnInXseyoDw3cRyxO30Y5chGbkFzw0nBlY"
)

verification_data_file = "/root/OpenRLHF/xuhao/verify/data/input/verification_data_test.json"
verification_result_file = "/root/OpenRLHF/xuhao/verify/data/output/verification_result_test_claude.json"
verification_system_message_file = "/root/OpenRLHF/xuhao/verify/data/input/verification_system_message.txt"
verification_few_shot_file = "/root/OpenRLHF/xuhao/verify/data/input/verification_few_shot.json"

def gpt_generate(messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=messages)
    return completion.choices[0].message.content.strip()

def generate_verification_result_gpt(num_verifications=1e8, num_few_shots=1, save_batch=5):
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
    
    start_idx = len(verification_result)
    end_idx = min(start_idx + num_verifications, len(verification_data))

    for idx, data in enumerate(verification_data[start_idx:end_idx]):
        print("Processing problem ", data["problem_index"], " step", data["step_index"])
        if idx % save_batch == 0 and idx != 0:
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
        
        verification_result.append({
                "problem_index": data["problem_index"],
                "step_index": data["step_index"],
                "verification_result": response
        })
    
    with open(verification_result_file, 'w') as f:
        json.dump(verification_result, f, indent=4)

if __name__ == "__main__":
    generate_verification_result_gpt()
