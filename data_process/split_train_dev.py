import random

'''
    This code is used to seperate the data into train-eval sets for pretraining
'''
# 设置随机种子以获得可重复的结果
random.seed(42)

'''
    corpus.txt obtained through command: 'cat env.txt soc.txt gov.txt | sort | uniq > corpus.txt'
'''
# 读取数据文件
with open('processed_data/filtered/corpus.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 随机打乱数据
random.shuffle(lines)

# 计算分割点
split_idx = int(0.9 * len(lines))

# 分割数据为训练集和评估集
train_lines = lines[:split_idx]
eval_lines = lines[split_idx:]

# 保存训练集
with open('processed_data/filtered/train.txt', 'w', encoding='utf-8') as file:
    file.writelines(train_lines)

# 保存评估集
with open('processed_data/filtered/eval.txt', 'w', encoding='utf-8') as file:
    file.writelines(eval_lines)

print(f"训练集数据行数: {len(train_lines)}")
print(f"评估集数据行数: {len(eval_lines)}")
