import json

'''
    用这个去拿E，S， G分类正确的和不正确的数据
'''


# 文件名列表
input_files = ['output/gov/combined.jsonl'] # get by command: 'cat gov/*.jsonl > gov/combined.jsonl'
output_true_file = 'output/gov/is_gov_true.jsonl'
output_false_file = 'output/gov/is_gov_false.jsonl'

# 打开输出文件
with open(output_true_file, 'w') as outfile_true, open(output_false_file, 'w') as outfile_false:
    # 遍历每个输入文件
    for file_name in input_files:
        # 打开输入文件
        with open(file_name, 'r') as infile:
            # 遍历文件中的每一行
            for line in infile:
                # 解析JSON数据
                data = json.loads(line)
                # 检查is_high_quality字段是否为false
                if (data.get('is_soc') is True) or (data.get('is_gov') is True) or (data.get('is_env') is True):
                    # 写入输出文件
                    json.dump(data, outfile_true)
                    outfile_true.write('\n')
                    outfile_true.flush()
                else:
                    json.dump(data, outfile_false)
                    outfile_false.write('\n')
                    outfile_false.flush()
