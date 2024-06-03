import json

# 文件路径
files = {
    'Env': 'output_9class/env/correct_data.jsonl',
    'Soc': 'output_9class/soc/correct_data.jsonl',
    'Gov': 'output_9class/gov/correct_data.jsonl'
}

def extract_data(file_path, label):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            # 根据对应的布尔字段添加数据
            if entry[f'is_{label.lower()}']:
                data.append({'text': entry['text'], 'label': label, 'sub_label': entry['label']})
    return data

# 提取并合并数据
all_data = []
for label, path in files.items():
    all_data.extend(extract_data(path, label))

# 保存到新的jsonl文件
output_file = 'nine_class_dataset/nineclass_esg_dataset.jsonl'
with open(output_file, 'w', encoding='utf-8') as file:
    for item in all_data:
        file.write(json.dumps(item) + '\n')

print(f"Data saved to {output_file} with {len(all_data)} items.")
