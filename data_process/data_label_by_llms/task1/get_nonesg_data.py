import json

# 文件名列表
input_files = ['processed_data/high_quality_data/chatgpt3/env_all_results.jsonl', 'processed_data/high_quality_data/chatgpt3/gov_all_results.jsonl', 'processed_data/high_quality_data/chatgpt3/soc_all_results.jsonl']
output_file = 'chatgpt3_nonesg.jsonl'

# 打开输出文件
with open(output_file, 'w') as outfile:
    # 遍历每个输入文件
    for file_name in input_files:
        # 打开输入文件
        with open(file_name, 'r') as infile:
            # 遍历文件中的每一行
            for line in infile:
                # 解析JSON数据
                data = json.loads(line)
                # 检查is_high_quality字段是否为false
                if data.get('is_high_quality') is False:
                    # 写入输出文件
                    json.dump(data, outfile)
                    outfile.write('\n')
