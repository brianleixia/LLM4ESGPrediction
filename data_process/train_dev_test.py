import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter
import os

# 读取数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# 分割数据，保持类别平衡的同时随机打乱
def balanced_stratified_split(data, train_size=0.81, val_size=0.09, test_size=0.1):
    # 提取标签用于分层抽样
    labels = [item['sub_label'] for item in data]

    # 创建分层抽样器实例
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=42)

    # 分割训练集和剩余部分（验证集+测试集）
    for train_indices, test_val_indices in sss.split(data, labels):
        train_data = [data[i] for i in train_indices]
        test_val_data = [data[i] for i in test_val_indices]
        test_val_labels = [labels[i] for i in test_val_indices]

    # 计算验证集占训练集和验证集总和的比例
    relative_val_size = val_size / (val_size + test_size)

    # 再次分割剩余部分为验证集和测试集
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=42)
    for val_indices, test_indices in sss_val_test.split(test_val_data, test_val_labels):
        val_data = [test_val_data[i] for i in val_indices]
        test_data = [test_val_data[i] for i in test_indices]

    return train_data, val_data, test_data

# 保存数据
def save_data(data, file_path):
    # 检查目录是否存在，如果不存在则创建
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 现在可以安全地写文件了
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

# 统计标签分布
def count_labels(data):
    labels = [item['sub_label'] for item in data]
    return Counter(labels)

# 主函数
def main():
    file_path = 'nine_class_dataset/nine_class_dataset.jsonl'
    data = load_data(file_path)
    train_data, val_data, test_data = balanced_stratified_split(data)

    # 保存分割后的数据集
    save_data(train_data, 'nine_class_dataset/stratified/81_09_10/train.jsonl')
    save_data(val_data, 'nine_class_dataset/stratified/81_09_10/val.jsonl')
    save_data(test_data, 'nine_class_dataset/stratified/81_09_10/test.jsonl')

    # 统计每个集合中每个标签的数量
    train_labels_count = count_labels(train_data)
    val_labels_count = count_labels(val_data)
    test_labels_count = count_labels(test_data)

    print("Label distribution in training set:", train_labels_count)
    print("Label distribution in validation set:", val_labels_count)
    print("Label distribution in testing set:", test_labels_count)

if __name__ == "__main__":
    main()
