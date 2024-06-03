import json
import os
import random
import re


def is_valid_text(text):
    # 检查文本中是否有连续的特殊符号
    if re.search(r"[^\w\s]{2,}", text):
        return False
    # 检查文本的词数是否少于20
    if len(text.split()) < 20:
        return False
    return True


def clean_text(text):
    # 删除不可见的Unicode字符
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    # 删除连续的特殊字符（非字母、数字、空格的字符）
    text = re.sub(r'[^a-zA-Z0-9\s]+', '', text)

    # 去除链接
    text = re.sub(r'http\S+', '', text)

    return text.strip()


def process_non_esg_relevant_jsonl(file_path, output_file, n_samples):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sampled_lines = random.sample(lines, min(n_samples, len(lines)))
        
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for line in sampled_lines:
            data = json.loads(line)
            input = clean_text(data["text"])

            sft_data_items = [{
                "instruction": "If the following text is ESG related data",
                "input": input,
                "output": data["explanation"],
            }, {
                "instruction": "Classify the following text into one of the four ESG categories: 'Env', 'Soc', 'Gov', or 'Non-ESG'.",
                "input": input,
                "output": "Label: 'Non-ESG'",
            }, {
                "instruction": "Classify the following text into one of the nine ESG categories: 'Climate Change', 'Natural Capital', 'Pollution and Waste', 'Human Capital', 'Product Liability', 'Community Relations', 'Corporate Governance', 'Business Ethics and Values', or 'Non-ESG'.",
                "input": input,
                "output": "Label: 'Non-ESG'",
            }]

            for item in sft_data_items:
                text_with_input = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
                item["text"] = text_with_input
            
                out_file.write(json.dumps(item, ensure_ascii=False) + '\n')
                out_file.flush()


def process_non_esg_txt(file_path, output_file, n_samples):
    valid_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text = line.strip()
            if text and is_valid_text(text):
                valid_lines.append(text)
    
    sampled_lines = random.sample(valid_lines, min(n_samples, len(valid_lines)))
    
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for text in sampled_lines:
            input = clean_text(text)

            sft_data_items = [{
                "instruction": "If the following text is ESG related data",
                "input": input,
                "output": "No",
            }, {
                "instruction": "Classify the following text into one of the four ESG categories: 'Env', 'Soc', 'Gov', or 'Non-ESG'.",
                "input": input,
                "output": "Label: 'Non-ESG'",
            }, {
                "instruction": "Classify the following text into one of the nine ESG categories: 'Climate Change', 'Natural Capital', 'Pollution and Waste', 'Human Capital', 'Product Liability', 'Community Relations', 'Corporate Governance', 'Business Ethics and Values', or 'Non-ESG'.",
                "input": input,
                "output": "Label: 'Non-ESG'",
            }]

            for item in sft_data_items:
                text_with_input = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
                item["text"] = text_with_input
            
                out_file.write(json.dumps(item, ensure_ascii=False) + '\n')
                out_file.flush()

# Paths to your files
non_esg_relevant_jsonl_path = '../processed_nonesg/non_esg_relevant.jsonl'
non_esg_txt_path = '../non_esg_data/non_esg.txt'
output_file = 'nonesg_sft_data_wtext.jsonl'

n_samples_jsonl = 8500
n_samples_txt = 5500

# Ensure the output file is empty before starting
open(output_file, 'w').close()

# Process each file with specified sample size
process_non_esg_relevant_jsonl(non_esg_relevant_jsonl_path, output_file, n_samples_jsonl)
process_non_esg_txt(non_esg_txt_path, output_file, n_samples_txt)

print("Non-ESG data has been processed with specified samples and saved.")
