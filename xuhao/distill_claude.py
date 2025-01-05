import json
from openai import OpenAI

client = OpenAI(
    base_url='https://api.feidaapi.com/v1',
    api_key="sk-rHxzHQ3cr9UIqufjXfiRqhW5OMd7GKd7YY1CS7CCYtpvEgE5"
)

def gpt_generate(messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=messages)
    return completion.choices[0].message.content.strip()

def generate_verification_result_gpt(num_problems=1e8, num_few_shots=1, save_batch=10):
    with open("xuhao/verification_dataset.json", 'r') as f:
        verification_data = json.load(f)
    with open("xuhao/verification_system_message.txt", 'r') as f:
        verification_system_message = f.read()
    with open("xuhao/verification_few_shot.json", 'r') as f:
        examples = json.load(f)
    with open("xuhao/claude_verification_result.json", 'r') as f:
        result = json.load(f)

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
    
    num_processed = len(result)
    end_idx = min(num_processed + num_problems, len(verification_data))
    for idx, data in enumerate(verification_data[num_processed:end_idx]):
        print(f"Processing problem {idx + num_processed}")
        if idx % save_batch == 0 and idx != 0:
            with open("xuhao/claude_verification_result.json", 'w') as f:
                json.dump(result, f, indent=4)
        tmp = []
        for idx, e in enumerate(data["input"]):
            print(f"Processing step {idx}")
            messages.append({
                "role": "user",
                "content": e
            })
            response = gpt_generate(messages=messages)
            
            tmp.append(response)
            messages.pop()

            if response.lower().startswith("evaluation: incorrect"):
                break
        
        result.append(tmp)
    
    with open("xuhao/claude_verification_result.json", 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    generate_verification_result_gpt()
