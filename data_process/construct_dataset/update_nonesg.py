import json
import re

def clean_text(text):
    # 删除不可见的Unicode字符
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    # 删除连续的特殊字符（非字母、数字、空格的字符）
    text = re.sub(r'[^a-zA-Z0-9\s]+', '', text)

    # 去除链接
    text = re.sub(r'http\S+', '', text)

    return text.strip()

input_file = 'non_esg_dataset.jsonl'
output_file = 'non_esg_dataset_updated.jsonl'

with open(input_file, 'r', encoding='utf-8') as input_jsonl, \
     open(output_file, 'w', encoding='utf-8') as output_jsonl:

    for line in input_jsonl:
        data = json.loads(line)
        data['text'] = clean_text(data['text'])  # 清洗"text"字段
        data['sub_label'] = data['label']
        updated_data = json.dumps(data, ensure_ascii=False)
        output_jsonl.write(updated_data + '\n')

print(f"已更新的数据已保存到 '{output_file}'")
