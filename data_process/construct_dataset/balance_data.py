import json
import random
import re

# 定义要下采样的标签及其目标样本数
target_sample_count = {
    'Natural Capital': 3000,
    'Climate Change': 3000,
    'Pollution and Waste': 3000,
    'Human Capital': 3000,
    'Community Relations': 3000,
    'Product Liability': 3000,
    'Corporate Governance': 3000,
    'Business Ethics and Values': 3000,
    'Non-ESG': 3000
}

input_file = 'combined.jsonl'
output_file = 'nine_class_dataset.jsonl'

def clean_text(text):
    # 删除不可见的Unicode字符
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    # 删除连续的特殊字符（非字母、数字、空格的字符）
    text = re.sub(r'[^a-zA-Z0-9\s]+', '', text)

    # 去除链接
    text = re.sub(r'http\S+', '', text)

    return text.strip()

# 创建一个字典来存储每个标签的样本
label_samples = {label: [] for label in target_sample_count}

# 读取数据并将每个样本分配到相应的标签列表中
with open(input_file, 'r', encoding='utf-8') as input_jsonl:
    for line in input_jsonl:
        data = json.loads(line)
        label = data['sub_label']
        data['text'] = clean_text(data['text'])  # 清洗"text"字段
        label_samples[label].append(data)

# 对每个标签的样本进行下采样，使其数量达到目标样本数
balanced_data = []
for label, samples in label_samples.items():
    if len(samples) > target_sample_count[label]:
        selected_samples = random.sample(samples, target_sample_count[label])
    else:
        selected_samples = samples
    balanced_data.extend(selected_samples)

# 随机打乱样本顺序
random.shuffle(balanced_data)

# 将平衡后的数据写入输出文件
with open(output_file, 'w', encoding='utf-8') as output_jsonl:
    for sample in balanced_data:
        output_jsonl.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"已平衡的数据已保存到 '{output_file}'")
