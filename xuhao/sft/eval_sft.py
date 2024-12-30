import json

def generate_solution_label_test():
    with open("xuhao/solve/data/output/solution_label.json", 'r') as f:
        solution_label = json.load(f)
    with open("xuhao/sft/data/input/sft_data.json", 'r') as f:
        sft_data_test = json.load(f)["test"]
    
    solution_label_test = []
    problem_index_test = set()
    for data in sft_data_test:
        problem_index_test.add(data["problem_index"])
    problem_index_test = sorted(problem_index_test)
    for idx in problem_index_test:
        solution_label_test.append(solution_label[idx])
    
    with open("xuhao/sft/data/input/solution_label_test.json", 'w') as f:
        json.dump(solution_label_test, f, indent=4)


if __name__ == "__main__":
    generate_solution_label_test()