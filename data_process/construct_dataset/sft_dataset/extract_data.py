from collections import defaultdict
import json
import random
import re

# 定义数据文件路径
data_file_path = 'esg_classification_sft_data.jsonl'

# 初始化按指令和类别分类的数据存储
data_by_instruction = defaultdict(list)

# 读取并分类数据

def determine_environmental_from_response(response):
    explanation = response['output']
    is_gov = explanation.lower().startswith("yes") or "answer: yes" in explanation.lower()
    return is_gov, explanation


with open(data_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        instruction = data['instruction']
        output = data['output']

        # 根据不同的指令和输出类别分类
        if "If the following text is ESG related data" in instruction:
            is_yes, _ = determine_environmental_from_response(data)
            category = "Yes" if is_yes else "No"
            # category = "Yes" if output.startswith("Yes") or "Answer: Yes" in output else "No"
        elif "Classify the following text into one of the four ESG categories" in instruction:
            label_match = re.search(r'Label: ([^\n\.]+)', output)
            label = label_match.group(1).strip() if label_match else "Unknown"
            category = label
        elif "Classify the following text into one of the nine ESG categories" in instruction:
            label_match = re.search(r'Label: ([^\n\.]+)', output)
            label = label_match.group(1).strip() if label_match else "Unknown"
            category = label
        else:
            continue  # 忽略不符合以上任一条件的数据
        
        # 将数据添加到对应的类别列表中
        data_by_instruction[(instruction, category)].append(data)

# 初始化最终的抽样数据列表
sampled_data = []

# 遍历每个类别，进行随机抽样
for (instruction, category), items in data_by_instruction.items():
    n_samples = 2000  # 根据指令决定抽样数量
    if len(items) > n_samples:
        sampled_items = random.sample(items, n_samples)
    else:
        sampled_items = items  # 如果不足目标数量，则全部选择
    sampled_data.extend(sampled_items)

# 打乱最终的抽样数据列表，以确保随机性
random.shuffle(sampled_data)

# 输出每个指令下的类别的数量
for (instruction, category), items in data_by_instruction.items():
    print(f"指令: {instruction}, 类别: {category}, 数量: {len(items)}")

# 定义输出文件路径
output_file_path = 'sampled_esg_data.json'

# 将抽样数据保存为JSON格式
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(sampled_data, output_file, ensure_ascii=False, indent=4)

print(f"Sampled and shuffled data saved to {output_file_path}.")
