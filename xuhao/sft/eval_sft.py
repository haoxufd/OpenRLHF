from tkinter import NO
from xuhao.utils import read_json_list_file, write_json_list_file
from xuhao.verify.calculate_verification_accuracy import get_verification_accuracy_plevel, get_verification_accuracy_slevel
import csv

def generate_solution_label_test(
        solution_label_file,
        sft_data_file,
        solution_label_test_file):
    """
    sft 阶段的 train 和 test 数据集, 其并集并不是全体数据集, 而只是 claude 蒸馏后 verification 正确的那部分
    所有 solution_label_file 中包含全部的数据, sft_data_file 中只包含一部分
    该函数作用是得到 sft test data 对应的 solution label 文件
    """
    solution_label = read_json_list_file(solution_label_file)
    sft_data_test = read_json_list_file(sft_data_file)["test"]
    
    solution_label_test = []
    problem_index_test = set()
    for data in sft_data_test:
        problem_index_test.add(data["problem_index"])
    problem_index_test = sorted(problem_index_test)
    for idx in problem_index_test:
        solution_label_test.append(solution_label[idx])
    
    write_json_list_file(solution_label_test_file, solution_label_test)

def postprocess_sft_verification_result(
        sft_verification_result_file_list: list, 
        sft_data_file: str):
    """
    SFT 阶段为了统计在 test 数据集上的 verification 准确率, 每隔一定 steps 就生成一次
    verification result, 这些结果存放在 xuhao/sft/data/output/eval_step_{i}.json 
    中, 但是其内容少了 problem_index 和 step_index
    该函数作用是加上这两项信息, 因为其中的数据顺序与 sft test data 中的数据是对应的, 所以
    遍历 sft test data 即可
    """
    sft_test_data = read_json_list_file(sft_data_file)["test"]

    for file in sft_verification_result_file_list:
        verification_result = read_json_list_file(file)
        
        new_verification_result = []
        for idx, data in enumerate(sft_test_data):
            new_verification_result.append({
                "problem_index": data["problem_index"],
                "step_index": data["step_index"],
                "verification_result": verification_result[idx]
            })
        
        write_json_list_file("verification_result_" + file, new_verification_result)

def analyze_verification_accuracy(
    verification_result_file_list: list, 
    solution_label_file: str, 
    ref_verification_result_file:str,
    plevel_csv_result_file,
    slevel_csv_result_file):
    """
    分析 SFT 训练过程中的 verification 准确率变化趋势, SFT 过程中测准确率是用 SFT Test Data 测的
    所以 solution_label_file 是这些 SFT Test Data 的 solution label
    ref_verification_result_file 是总的 GPT 蒸馏结果
    """
    plevel_result = []
    slevel_result = []
    for file in verification_result_file_list:
        plevel_result.append(get_verification_accuracy_plevel(file, solution_label_file))
        slevel_result.append(get_verification_accuracy_slevel(file, ref_verification_result_file))
    
    for data in plevel_result:
        total = sum(data)
        right = data[0] + data[3]
        acc = right / total
        data.insert(0, acc)
        data.insert(0, right)
        data.insert(0, total)
    
    for data in slevel_result:
        total = sum(data)
        right = data[0] + data[3]
        acc = right / total
        data.insert(0, acc)
        data.insert(0, right)
        data.insert(0, total)
    
    plevel_result.insert(0, ["Num Total", "Num Right", "Acc", "CC", "CI", "IC", "II"])
    slevel_result.insert(0, ["Num Total", "Num Right", "Acc", "CC", "CI", "IC", "II"])

    with open(plevel_csv_result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(plevel_result)
    
    with open(slevel_csv_result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(slevel_result)

if __name__ == "__main__":
    # generate_solution_label_test(
    #     solution_label_file="xuhao/solve/data/output/solution_label_new.json",
    #     sft_data_file="xuhao/sft/data/input/sft_data_new.json",
    #     solution_label_test_file="xuhao/sft/data/input/solution_label_test_new.json"
    # )
    files = [f"xuhao/sft/data/output/eval_step_{i}" for i in range(16)]
    postprocess_sft_verification_result(
        sft_verification_result_file_list=files, 
        sft_data_file="xuhao/sft/data/input/sft_data_new.json")