import json
from collections import Counter

# 初始化一个计数器
label_counts = Counter()

# 打开JSONL文件并逐行读取
with open("gov/combined.jsonl", "r") as file:
    for line in file:
        # 将每行的JSON字符串转换为字典
        record = json.loads(line)
        # 提取label字段的值，如果label不存在则默认为空字符串
        label = record.get("label", "")
        # 更新计数器
        label_counts[label] += 1

# 打印每个label及其出现的次数
for label, count in label_counts.items():
    print(f"Label '{label}': {count} times")


with open("gov_label_counts.txt", "w") as output_file:
    for label, count in label_counts.items():
        output_file.write(f"Label '{label}': {count} times\n")
