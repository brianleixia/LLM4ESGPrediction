import json
import re

'''
    用于处理merged_nonesg_data中的explanation的Relevance字段的数据，找到与ESG无关的数据即为Non ESG data
'''
# 定义函数用于判断explanation中Relevance的相关性
def check_relevance(explanation):
    # 使用正则表达式查找Relevance后的描述
    match = re.search(r"relevance: (.+)", explanation, re.IGNORECASE)
    if match:
        relevance_desc = match.group(1).lower()  # 将匹配到的描述转换为小写
        # 检查是否包含"no"或"not"
        if " no " in relevance_desc or " not " in relevance_desc:
            return "non_esg"
        else:
            return "esg"
    return "ambiguous"

# 打开原始文件并读取数据
with open("merged_nonesg_data.jsonl", "r") as file:
    lines = file.readlines()

# 初始化分类数据容器
non_esg_data = []
esg_data = []
ambiguous_data = []

# 遍历每一行数据
for line in lines:
    record = json.loads(line)
    explanation = record.get("explanation", "")
    category = check_relevance(explanation)

    # 根据相关性分类
    if category == "non_esg":
        non_esg_data.append(record)
    elif category == "esg":
        esg_data.append(record)
    else:
        ambiguous_data.append(record)

# 定义函数用于将数据保存到文件
def save_data(data, filename):
    with open(filename, "w") as file:
        for record in data:
            file.write(json.dumps(record) + "\n")

# 保存分类后的数据到不同的文件
save_data(non_esg_data, "processed_nonesg/non_esg_relevant.jsonl")
save_data(esg_data, "processed_nonesg/esg_relevant.jsonl")
save_data(ambiguous_data, "processed_nonesg/esg_ambiguous.jsonl")
