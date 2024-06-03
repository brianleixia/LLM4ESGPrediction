import json
import random
import re

# 设置随机抽取的数据量
n_samples_jsonl = 8500
n_samples_txt = 5500

def load_and_sample_jsonl(file_path, n_samples):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sampled_lines = random.sample(lines, min(n_samples, len(lines)))
        sampled_data = [json.loads(line)['text'] for line in sampled_lines]
    return sampled_data

def load_and_sample_txt(file_path, n_samples, min_words=8):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
        valid_lines = [line for line in lines if len(re.findall(r'\w+', line)) >= min_words]
        sampled_lines = random.sample(valid_lines, min(n_samples, len(valid_lines)))
    return sampled_lines

# 加载和抽样数据
jsonl_data = load_and_sample_jsonl('processed_nonesg/non_esg_relevant.jsonl', n_samples_jsonl)
txt_data = load_and_sample_txt('non_esg_data/non_esg.txt', n_samples_txt)

# 合并数据并添加标签
combined_data = [{'text': text, 'label': 'Non-ESG'} for text in jsonl_data + txt_data]

# 保存为jsonl格式
output_file = 'non_esg_dataset.jsonl'
with open(output_file, 'w', encoding='utf-8') as file:
    for item in jsonl_data:
        file.write(json.dumps(item) + '\n')

print(f"Data saved to {output_file} with {len(jsonl_data)} items.")
