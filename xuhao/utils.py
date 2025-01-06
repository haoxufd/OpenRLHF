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
    with open("xuhao/verification_dataset.json", 'r') as f:
        verification_data = json.load(f)
    
    result = []

    for data in verification_data:
        if data["problem index"] >= 7162:
            result.append(data)
    
    with open("xuhao/verification_dataset_eval.json", 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()