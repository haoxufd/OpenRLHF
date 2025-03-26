import json
from itertools import groupby
from datasets import interleave_datasets, load_dataset
import re
from deepspeed import get_accelerator
from sympy import denom
import torch

def print_item_num(json_file):
    with open(json_file, 'r') as f:
        print(len(json.load(f)))

def read_json_list_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def write_json_list_file(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def get_redistill_problem(solution_file, old_solution_file):
    """
    获取需要重蒸馏的数据, solution 中跟 old_solution 重复的不需要重新蒸馏, 反正需要重新蒸馏
    返回一个 hash table, 长度为 problem 总数, 置为 True 的是需要重新蒸馏的
    """
    solution = read_json_list_file(solution_file)
    old_solution = read_json_list_file(old_solution_file)
    assert len(solution) == len(old_solution)

    num = len(solution)
    ret = [False] * num
    cnt = 0
    for i in range(num):
        if solution[i]["solution"] != old_solution[i]["solution"]:
            ret[i] = True
            cnt += 1
    print(cnt)
    return ret

def get_grouped_data(data):
    """
    convert format like this:
    [
        {"problem_index": 0, "step_index": 0},
        {"problem_index": 0, "step_index": 1},
        {"problem_index": 1, "step_index": 0},
        {"problem_index": 1, "step_index": 1}
    ]
    to this:
    [
        [
            {"problem_index": 0, "step_index": 0},
            {"problem_index": 0, "step_index": 1}
        ],
        [
            {"problem_index": 1, "step_index": 0},
            {"problem_index": 1, "step_index": 1}
        ]
    ]
    """
    return [list(group) for key, group in groupby(data, key=lambda x: x["problem_index"])]

def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        
        data = load_dataset(dataset, "main")
        strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)
        
        # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset

def parse_solution(solution: str):
    value = re.findall(r'#### ([+-]?[\d,]*\.?[\d,]+)', solution)
    value = value[0].replace(',', '') if len(value) > 0 else None
    return float(value) if value is not None else None

def get_solve_result(solutions: list, ref_solutions: list):
    """
    由于是通过 batch inference 得到的 solution, 其中的数据可能多于 ref_solution
    但两者数据是按顺序对应的, 遍历的时候以 ref_solution 为主即可
    """
    num_correct = num_incorrect = 0
    incorrect_indices = []
    for i in range(len(ref_solutions)):
        value = parse_solution(solutions[i])
        ref_value = parse_solution(ref_solutions[i])

        if value is None:
            num_incorrect += 1
            incorrect_indices.append(i)
        elif value is not None and value != ref_value:
            num_incorrect += 1
            incorrect_indices.append(i)
        else:
            num_correct += 1

    total = num_correct + num_incorrect
    return [num_correct, num_incorrect, total, num_correct / total, incorrect_indices]

def get_steps(solution: str):
    """
    solution 的格式一定是正确的
    """
    steps = solution.strip().split('<|reserved_special_token_0|>')
    # steps[-1] = <|eot_id|>
    steps = steps[:-1]
    # steps[-1] 以 #### FINAL_VALUE 结束, 需要去掉这个后缀, 因为 reward model 训练的时候最后一步没带后缀
    steps[-1] = steps[-1].split('\n')[0]
    # 去掉 step 最后的 \n, 同样是因为 reward model 训练的时候 step 没有以 \n 结束
    steps = [step.strip() for step in steps]
    return steps

def str_to_num(num_str: str):
    if '/' not in num_str:
        return float(num_str.replace(',', ''))
    else:
        numerator = float(num_str.split('/')[0].replace(',', ''))
        denominator = float(num_str.split('/')[1].replace(',', ''))
        if denominator == 0:
            return None
        return numerator / denominator

def get_final_value_from_solution(solution: str) -> float | None:
    """
    solution 的格式一定是正确的
    """
    end_mark = "<|reserved_special_token_0|><|eot_id|>"
    number_content = solution.split("#### ")[-1][:-len(end_mark)]
    assert ' ' not in number_content
    return str_to_num(number_content)

def get_final_value_from_ref_solution(ref_solution: str) -> float:
    number_content = ref_solution.split("#### ")[-1]
    return float(number_content.replace(',', ''))

def find_newline_indices(s):
    indices = []
    for i, char in enumerate(s):
        if char == '\n':
            # 如果是第一个 '\n' 或与前一个字符不连续
            if i == 0 or s[i - 1] != '\n':
                indices.append(i)
    return indices

def is_number(text: str) -> bool:
    pattern = r'^-?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\d+/\d+|\.\d+)$'
    match = re.fullmatch(pattern, text)
    return match is not None

def group_elements(elements, group_sizes):
    """
    将列表 elements 中的元素按照 group_sizes 中指定的分组大小分组.

    参数:
        elements (list): 待分组的元素列表, 其长度必须等于 sum(group_sizes).
        group_sizes (list of int): 每组需要的元素个数列表, 总和应与 elements 的长度相等.

    返回:
        list of list: 分组后的列表, 每个子列表的长度对应 group_sizes 中的值.

    示例:
        >>> elements = [1, 2, 3, 4, 5, 6]
        >>> group_sizes = [2, 1, 3]
        >>> group_elements(elements, group_sizes)
        [[1, 2], [3], [4, 5, 6]]
    """
    if len(elements) != sum(group_sizes):
        raise ValueError("元素总数必须等于 group_sizes 中数字的和.")
    
    result = []
    start_index = 0
    for size in group_sizes:
        # 切片得到当前组
        group = elements[start_index : start_index + size]
        result.append(group)
        start_index += size

    return result

def solution_is_valid(solution:str):
    pattern = (
        r'^'
        r'(?:.+\n<\|reserved_special_token_0\|>)*'
        r'.+\n'
        r'####\s'
        r'-?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\d+/\d+|\.\d+)'
        r'<\|reserved_special_token_0\|><\|eot_id\|>'
        r'$'
    )
    return re.fullmatch(pattern, solution) is not None

def get_eostep_indices(response_sequences: list[list[int]], step_split_token_id: int)->list[list[int]]:
    """
    获取每个响应序列中步骤分割符的索引.

    本函数旨在找出在每个响应序列中, 每个步骤结束（由特定的分割符标识）的位置索引.
    这对于处理分步骤的响应数据特别有用, 比如在分析对话系统中每个回复包含的步骤时.

    参数:
    response_sequences (list[list[int]]): 一个二维列表, 包含多个响应序列, 每个响应序列是由整数表示的token id序列.
    step_split_token_id (int): 用于标识每个步骤结束的特定token id.

    返回:
    list[list[int]]: 一个二维列表, 每个子列表包含对应响应序列中每个步骤结束的token id索引.
    """
    # 初始化用于存储所有响应序列中步骤分割符索引的列表
    eostep_indices = []

    # 遍历每个响应序列
    for response in response_sequences:
        # 初始化用于存储当前响应序列中步骤分割符索引的列表
        indices = []
        # 从响应序列的起始位置开始查找
        start = 0
        # 当前查找位置未超出响应序列长度时, 继续查找
        while start < len(response):
            try:
                # 查找 step_split_token_id 在 response 中的位置
                idx = response.index(step_split_token_id, start)
                # 将找到的位置索引添加到列表中
                indices.append(idx)
                # 更新下一次查找的起始位置
                start = idx + 1
            except ValueError:
                # 如果未找到 step_split_token_id, 则退出循环
                break
        # 将当前响应序列中所有步骤分割符的索引添加到结果列表中
        eostep_indices.append(indices)

    # 返回所有响应序列中步骤分割符的索引列表
    return eostep_indices

def preallocate_memory():
    # 获取当前GPU的显存总量
    total_mem = get_accelerator().total_memory()
    
    # 预留 10% 的显存作为安全余量，防止OOM
    reserve_mem = int(total_mem * 0.03)
    prealloc_size = total_mem - reserve_mem
    
    # 分配一个大的张量占用显存（保留引用防止被回收）
    dummy_tensor = torch.empty(
        (prealloc_size // 4,),  # 假设float32占4字节
        dtype=torch.float32,
        device=torch.cuda.current_device()
    )
    return dummy_tensor

def analyze_training_logs(log_file_path, output_dir, debug = False):
    """
    分析训练日志文件，计算每个训练步骤的准确率，并生成CSV文件和图表。
    
    参数:
    log_file_path (str): 日志文件路径
    output_dir (str): 输出目录，用于保存CSV文件和图表
    debug (bool): 是否输出调试信息
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取日志文件行
    with open(log_file_path, 'r', encoding='utf-8') as file:
        log_lines = file.readlines()
    
    if debug:
        print(f"日志文件共 {len(log_lines)} 行")
    
    # 按Make Experience编号分组
    experiences = []
    current_exp_num = None
    current_exp_lines = []
    
    for line in log_lines:
        exp_match = re.search(r'Make \'Experience\' (\d+)', line)
        if exp_match:
            # 如果已经在处理一个experience，保存它
            if current_exp_num is not None:
                experiences.append((current_exp_num, current_exp_lines))
            
            # 开始新的experience
            current_exp_num = int(exp_match.group(1))
            current_exp_lines = [line]
        elif current_exp_num is not None:
            # 继续添加到当前experience
            current_exp_lines.append(line)
    
    # 添加最后一个experience
    if current_exp_num is not None and current_exp_lines:
        experiences.append((current_exp_num, current_exp_lines))
    
    if debug:
        print(f"找到 {len(experiences)} 个Experience记录")
    
    # 按步骤分组
    steps_data = {}
    current_step = 1
    
    for exp_num, exp_lines in experiences:
        # 检查是否是新步骤的开始
        if exp_num == 0 and len(steps_data.get(current_step, [])) > 0:
            current_step += 1
        
        if current_step not in steps_data:
            steps_data[current_step] = []
        
        steps_data[current_step].append((exp_num, exp_lines))
    
    # 过滤掉 Experience 数量不足 32 的步骤
    filtered_steps_data = {step: exps for step, exps in steps_data.items() if len(exps) >= 32}

    if debug:
        print(f"找到 {len(steps_data)} 个训练步骤")
        # for step, exps in steps_data.items():
        #     print(f"步骤 {step}: {len(exps)} 个Experience")
        
        # 打印被忽略的步骤
        ignored_steps = set(steps_data.keys()) - set(filtered_steps_data.keys())
        if ignored_steps:
            print(f"以下步骤因 Experience 数量不足 32 被忽略: {sorted(ignored_steps)}")

    # 使用过滤后的步骤数据进行后续处理
    steps_data = filtered_steps_data

    if debug:
        print(f"保留 {len(steps_data)} 个训练步骤")

    results_actor = []
    results_reward = []

    # 处理每个步骤
    for step_num, experiences in steps_data.items():
        if debug:
            print(f"处理步骤 {step_num}...")
        
        # 初始化计数器
        total_correct = 0
        total_samples = 0
        
        # rm acc计算的变量
        true_positive = 0  # T-T 和 F-F
        total_samples_rm = 0
        
        # 遍历每个experience
        for exp_idx, (exp_num, exp_lines) in enumerate(experiences):
            # 提取 Solution labels
            solution_labels = []
            label_found = False
            
            for i, line in enumerate(exp_lines):
                if "Solution labels>>>>>>" in line and i+1 < len(exp_lines):
                    next_line = exp_lines[i+1]
                    label_match = re.search(r'\[(.*?)\]', next_line)
                    if label_match:
                        label_str = label_match.group(1)
                        try:
                            # 使用安全的列表解析来解析布尔值
                            solution_labels = [s.strip().lower() == 'true' for s in label_str.split(',')]
                            label_found = True
                            break
                        except Exception as e:
                            if debug:
                                print(f"解析Solution labels出错: {e}")
            
            if not label_found:
                if debug and exp_idx < 2:
                    print(f"Experience {exp_num} 无法找到Solution labels")
                continue
                
            # if debug and exp_idx < 2:
            #     print(f"Experience {exp_num} 的Solution labels: {solution_labels}")
            
            # 提取 Verification result
            verification_results = []
            verif_found = False
            
            for i, line in enumerate(exp_lines):
                if "Verification result after postprocessing>>>>>>" in line and i+1 < len(exp_lines):
                    next_line = exp_lines[i+1]
                    
                    # 直接提取行中的嵌套列表
                    try:
                        # 尝试提取完整的嵌套列表字符串
                        verif_str_match = re.search(r'\](.*?)$', next_line)
                        if verif_str_match:
                            verif_str = verif_str_match.group(1).strip()
                            
                            # 确保字符串是有效的Python列表格式
                            if not verif_str.startswith("[") or not verif_str.endswith("]"):
                                verif_str = "[" + verif_str + "]"
                            
                            # 替换True/False为Python的true/false
                            verif_str = verif_str.replace("True", "true").replace("False", "false")
                            
                            # 使用eval安全地解析嵌套列表
                            parsed_list = eval(verif_str, {"true": True, "false": False})
                            
                            # 将每个内部列表转换为布尔值列表
                            for inner_list in parsed_list:
                                if isinstance(inner_list, list):
                                    verification_results.append([bool(item) for item in inner_list])
                            
                            verif_found = True
                            break
                    except Exception as e:
                        if debug:
                            print(f"解析Verification result出错: {e}")
                            print(f"问题行: {next_line}")
            
            if not verif_found:
                if debug and exp_idx < 2:
                    print(f"Experience {exp_num} 无法找到Verification result")
                continue
                
            # if debug and exp_idx < 2:
            #     print(f"Experience {exp_num} 的Verification results: {verification_results}")
            
            # 确保两个列表长度相同
            if len(solution_labels) != len(verification_results):
                if debug and exp_idx < 2:
                    print(f"Experience {exp_num} Solution labels和Verification result长度不匹配: {len(solution_labels)} vs {len(verification_results)}")
                continue
            
            # 更新计数
            total_correct += sum(solution_labels)
            total_samples += len(solution_labels)
            
            # 统计rm acc
            for i, (label, verification) in enumerate(zip(solution_labels, verification_results)):
                if label:  # Solution label is True
                    if all(verification):  # All verification steps are True
                        true_positive += 1
                else:  # Solution label is False
                    if not all(verification):  # At least one verification step is False
                        true_positive += 1
                total_samples_rm += 1
        
        
        # 计算acc和rm acc
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        rm_accuracy = true_positive / total_samples_rm if total_samples_rm > 0 else 0
        
        results_actor.append({
            'Num Correct':total_correct,
            'Num Inorrect':total_samples - total_correct,
            'Total':total_samples,
            'Acc': accuracy,
        })
        results_reward.append({
            'Num Correct': true_positive,
            'Num Inorrect':total_samples_rm - true_positive,
            'Total':total_samples_rm,
            'Acc': rm_accuracy
        })
    
    # 保存CSV文件
    df_actor = pd.DataFrame(results_actor)
    actor_csv_path = os.path.join(output_dir, 'results_actor.csv')
    df_actor.to_csv(actor_csv_path, index=False)
    
    df_reward = pd.DataFrame(results_reward)
    reward_csv_path = os.path.join(output_dir, 'results_reward.csv')
    df_reward.to_csv(reward_csv_path, index=False)

    # 生成图表
    plt.figure(figsize=(12, 6))
    
    # am曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1,len(df_actor) + 1), df_actor['Acc'], 'b-', marker='o')
    plt.title('Actor Model Accuracy with Step')
    plt.xlabel('Step')
    plt.ylabel('AM Accuracy')
    plt.grid(True)
    
    # rm曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1,len(df_reward) + 1), df_reward['Acc'], 'r-', marker='o')
    plt.title('Reward Model Accuracy with Step')
    plt.xlabel('Step')
    plt.ylabel('RM Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'training_metrics_plot.png')
    plt.savefig(plot_path)
    
    # 关闭图表
    plt.close()
    
    return actor_csv_path, reward_csv_path, plot_path