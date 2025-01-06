import json

def generate_problem_desc():
    desc = []
    with open("xuhao/verification_dataset.json", 'r') as f:
        obj = json.load(f)
    
    acc = 0
    for idx, v in enumerate(obj):
        desc.append({
            "index": idx,
            "num steps": len(v["input"]),
            "acc": acc})
        acc += len(v["input"])
    
    with open("xuhao/problem_desc.json", 'w') as f:
        json.dump(desc, f, indent=4)

def main():
    with open("xuhao/solution_dataset.txt", 'r') as f:
        solution_data = f.readlines()
    
    result = []
    for idx, data in enumerate(solution_data):
        dict_data = {"index": idx}
        for k, v in json.loads(data).items():
            dict_data[k] = v
        result.append(dict_data)

    with open("xuhao/solution_dataset.json", 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()