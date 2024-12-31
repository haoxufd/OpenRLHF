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
    pass

if __name__ == "__main__":
    generate_problem_desc()