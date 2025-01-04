import json
import re
from openai import OpenAI

def gen_statement_label():
    client = OpenAI()
    statement_label = []
    with open("xuhao/verify_prompt.txt", 'r') as f:
        verification_developer_message = f.read()

    with open("xuhao/verification_dataset.json", 'r') as f:
        obj = json.load(f)
        for q in obj:
            tmp = []
            go_on = True
            for s in q:
                if go_on:
                    messages = [
                        {"role": "developer", "content": verification_developer_message},
                        {"role": "user", "content": s}
                    ]
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages
                    )
                    response = completion.choices[0].message.content
                    tmp.append(response)
                    # TODO: Handle case when response is None
                    if response is None:
                        raise TypeError()
                    elif "incorrect" in response.split('\n')[0].lower():
                        go_on = False
                else:
                    tmp.append("")
            statement_label.append(tmp)


def gen_verification_dataset_and_answer_label():
    data = []
    answer_label = []
    with open("xuhao/result.txt", 'r') as f:
        for idx, line in enumerate(f.readlines()):
            obj = json.loads(line)
            ref_answer = re.findall(r'#### ([+-]?\d*\.?\d+)', obj["reference answer"])
            answer = re.findall(r'#### ([+-]?\d*\.?\d+)', obj["answer"])

            assert len(ref_answer) == 1
            assert len(answer) < 2
            if len(answer) == 0:
                # cannot get an result value, ends with last step, not "#### xxx"
                # this kind of answer is definitely wrong, since reference answer 
                # can always produce a final value
                answer_label.append(False)
                steps = obj["answer"].split('\n')
            else:
                # length = 1
                answer_label.append(float(ref_answer[0]) == float(answer[0]))
                steps = obj["answer"].split('\n')[:-1]

            tmp = {"input": []}
            for i in range(len(steps)):
                one_generate = "Problem:\n{}\n\nReference Solution:\n{}\n\nPrevious Steps:\n{}\n\nStep to Evaluate:\n{}".format(obj["question"], obj["reference answer"], '\n'.join(steps[:i]), steps[i])
                tmp["input"].append(one_generate)
            data.append(tmp)
    
    with open("xuhao/verification_dataset.json", 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    with open("xuhao/answer_label.json", 'w', encoding="utf-8") as f:
        json.dump(answer_label, f, indent=4)

def gen_answer_label():
    answer_label = []
    with open("xuhao/result.txt", 'r') as f:
        for idx, line in enumerate(f.readlines()):
            obj = json.loads(line)
            # reference answer may take the form as "10,800"
            ref_answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', obj["reference answer"])
            ref_answer[0] = ref_answer[0].replace(',', '')
            answer = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', obj["answer"])

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
    
    with open("xuhao/answer_label.json", 'w', encoding="utf-8") as f:
        json.dump(answer_label, f, indent=4)

if __name__ == "__main__":
    gen_answer_label()