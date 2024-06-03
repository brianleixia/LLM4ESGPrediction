import json

# 文件名列表
input_files = [
    'processed_data/high_quality_data/qwen/env_high_quality_data.jsonl',
    'processed_data/high_quality_data/chatgpt3/env_high_quality_data.jsonl', 
    'processed_data/high_quality_data/glm/env_high_quality_data.jsonl', 
]
output_file = 'merged_env_hqdata.jsonl'

# 存储text字段的集合
texts = set()

# 存储最终的JSON对象
merged_data = []

# 处理基准文件（file1.jsonl）
with open(input_files[0], 'r') as base_file:
    for line in base_file:
        data = json.loads(line)
        text = data.get('text')
        if text not in texts:
            texts.add(text)
            merged_data.append(data)

# 处理其他文件
for file_name in input_files[1:]:
    with open(file_name, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            text = data.get('text')
            if text not in texts:
                texts.add(text)
                merged_data.append(data)

# 写入合并后的数据到新文件
with open(output_file, 'w') as outfile:
    for data in merged_data:
        json.dump(data, outfile)
        outfile.write('\n')
