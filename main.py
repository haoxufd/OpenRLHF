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
    with open("xuhao/verify/data/output/verification_result_train_claude.json", 'r') as f:
       result_train = json.load(f)
    
    with open("xuhao/verify/data/output/verification_result_test_claude.json", 'r') as f:
       result_test = json.load(f)
    
    new_result_test = []

    for data in result_test:
        data["problem_index"] += 7473
        if data["verification_result"] != "NIL":
            new_result_test.append(data)

    final_result = result_train + new_result_test
    for data in final_result:
        verification_result = data["verification_result"].strip()
        assert verification_result.endswith("Evaluation Result: INCORRECT") or verification_result.endswith("Evaluation Result: CORRECT")
        
    with open("xuhao/verify/data/output/verification_result_claude.json", 'w') as f:
        json.dump(final_result, f, indent=4)
