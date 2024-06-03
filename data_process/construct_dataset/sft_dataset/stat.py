from collections import Counter
import json
import re

# 根据文件扩展名自动判断文件类型并适当读取数据
def read_data(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            data_array = json.load(file)
            for data in data_array:
                yield data
    else:
        raise ValueError("Unsupported file format. Please use either .json or .jsonl files.")

data_file_path = 'sampled_esg_data.json'  # Update to your file path

# 初始化计数器
yes_no_counter = Counter()
four_categories_counter = Counter()
nine_categories_counter = Counter()

# 定义从响应中确定环境的函数
def determine_environmental_from_response(response):
    explanation = response['output']
    is_gov = explanation.lower().startswith("yes") or "answer: yes" in explanation.lower()
    return is_gov, explanation

# 使用新的逻辑处理Yes/No判断，四个类别的判断，九个类别的判断
for data in read_data(data_file_path):
    instruction = data['instruction']
    output = data['output']

    if instruction.startswith("If the following text is ESG related data"):
        is_yes, _ = determine_environmental_from_response(data)
        result = "Yes" if is_yes else "No"
        yes_no_counter[result] += 1
    elif instruction.startswith("Classify the following text into one of the four ESG categories"):
        label_match = re.search(r'Label: ([^\n\.]+)', output)
        label = label_match.group(1).strip() if label_match else "Unknown"
        four_categories_counter[label] += 1
    elif instruction.startswith("Classify the following text into one of the nine ESG categories"):
        label_match = re.search(r'Label: ([^\n\.]+)', output)
        label = label_match.group(1).strip() if label_match else "Unknown"
        nine_categories_counter[label] += 1

# 打印统计结果
print("Yes/No Distribution:", yes_no_counter)
print("\nFour Categories Distribution:", four_categories_counter)
print("\nNine Categories Distribution:", nine_categories_counter)

# 保存统计结果
results_file_path = 'output_statistics.json'
with open(results_file_path, 'w', encoding='utf-8') as file:
    json.dump({
        "Yes/No Distribution": yes_no_counter,
        "Four Categories Distribution": four_categories_counter,
        "Nine Categories Distribution": nine_categories_counter
    }, file, ensure_ascii=False, indent=4)

print(f"\nStatistics saved to {results_file_path}.")
