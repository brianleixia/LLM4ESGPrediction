import sys
import json
import re

def extract_label(predicted_label):
    # 匹配"Label:"后的任意文本，直到字符串结束
    match = re.match(r'^Label:\s*(.+)$', predicted_label, re.IGNORECASE)
    return match.group(1).strip() if match else predicted_label

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            item = json.loads(line)
            # 使用extract_label函数处理predicted_label
            # 无论是否包含"Label: "，都将处理后的predicted_label作为响应存储
            item['response'] = item['predicted_label']
            item['predicted_label'] = extract_label(item['predicted_label'])


            json.dump(item, outfile)
            outfile.write('\n')

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    process_file(input_file_path, output_file_path)
