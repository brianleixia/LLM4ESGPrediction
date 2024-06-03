import json

# 文件路径
files = {
    'Env': 'output/env/is_env_true.jsonl',
    'Soc': 'output/soc/is_soc_true.jsonl',
    'Gov': 'output/gov/is_gov_true.jsonl'
}

def extract_data(file_path, label):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            # 根据对应的布尔字段添加数据
            if entry[f'is_{label.lower()}']:
                data.append({'text': entry['text'], 'label': label})
    return data

# 提取并合并数据
all_data = []
for label, path in files.items():
    all_data.extend(extract_data(path, label))

# 保存到新的jsonl文件
output_file = 'fourclass_esg_dataset.jsonl'
with open(output_file, 'w', encoding='utf-8') as file:
    for item in all_data:
        file.write(json.dumps(item) + '\n')

print(f"Data saved to {output_file} with {len(all_data)} items.")
