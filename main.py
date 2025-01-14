import json


if __name__ == "__main__":
    with open("xuhao/solve/data/output/solution.json", 'r') as f:
        solution = json.load(f)
    
    for data in solution:
        if "####" not in data["solution"]:
            print(data["index"])
