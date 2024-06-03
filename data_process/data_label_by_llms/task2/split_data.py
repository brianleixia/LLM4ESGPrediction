import json
import os

def split_jsonl_file(file_path, output_folder, num_splits=5):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取并加载整个JSONL文件
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 计算每个分割文件的大小
    split_size = len(lines) // num_splits
    for i in range(num_splits):
        split_lines = lines[i * split_size: (i + 1) * split_size]
        
        # 处理最后一份数据以包含所有剩余行
        if i == num_splits - 1:
            split_lines = lines[i * split_size:]

        # 构造输出文件路径
        output_file_path = os.path.join(output_folder, f"split_{i}.jsonl")

        # 保存分割后的数据
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(split_lines)

# 使用此函数分割文件
file_paths = ["merged_env_hqdata.jsonl", "merged_soc_hqdata.jsonl", "merged_gov_hqdata.jsonl"]
output_folders = ["split_data/env", "split_data/gov", "split_data/soc"]

for file_path, output_folder in zip(file_paths, output_folders):
    split_jsonl_file(file_path, output_folder)
