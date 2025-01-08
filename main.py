import json

def find_duplicate_questions(data):
    # 创建字典存储问题和对应的索引
    question_indices = {}
    
    # 遍历数据，将相同问题的索引存储在列表中
    for i, item in enumerate(data):
        question = item['question']
        if question in question_indices:
            question_indices[question].append(i)
        else:
            question_indices[question] = [i]
    
    # 找出有重复的问题，组成元组对
    duplicate_pairs = []
    for indices in question_indices.values():
        if len(indices) > 1:
            # 为每个重复问题创建所有可能的索引对
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    duplicate_pairs.append((indices[i], indices[j]))
    
    # 按第一个索引排序
    duplicate_pairs.sort()
    
    return duplicate_pairs

if __name__ == "__main__":
    with open("xuhao/verify/data/output/verification_result_all_claude.json", 'r') as f:
       result = json.load(f)
    
    for data in result:
        verification = data["verification_result"].strip()
        tmp = verification.split("\n\n")

        assert tmp[0].strip() in ("Evaluation: INCORRECT", "Evaluation: CORRECT")

        data["verification_result"] = "\n\n".join(tmp[1:]) + "\n\nEvaluation Result: " + tmp[0].strip().split(' ')[-1]
    
    with open("xuhao/verify/data/output/new_verification_result_all_claude.json", 'w') as f:
        json.dump(result, f, indent=4)
