import json

# 创建存储正确数据的JSONL文件和存储错误数据的JSONL文件
correct_output_file = 'correct_data.jsonl'
incorrect_output_file = 'incorrect_data.jsonl'

# 创建一个字典来映射正确的标签
# soc: Human Capital, Product Liability, and Community Relations.
# gov: Corporate Governance, and Business Ethics and Values
correct_labels = {
    'Corporate Governance': 1,
    'Business Ethics and Values': 1,
}

# 打开源文件和两个输出文件
with open('gov/combined.jsonl', 'r', encoding='utf-8') as input_file, \
     open(correct_output_file, 'w', encoding='utf-8') as correct_output, \
     open(incorrect_output_file, 'w', encoding='utf-8') as incorrect_output:

    for line in input_file:
        data = json.loads(line)
        
        # 检查标签是否属于正确的类别
        if data['label'] in correct_labels:
            # 将正确的数据写入正确的输出文件
            correct_output.write(json.dumps(data, ensure_ascii=False) + '\n')
        else:
            # 将错误分类的数据写入错误的输出文件
            incorrect_output.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"正确分类的数据已保存到 '{correct_output_file}'")
print(f"错误分类的数据已保存到 '{incorrect_output_file}'")
