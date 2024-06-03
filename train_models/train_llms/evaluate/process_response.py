import json
import re
import os
import argparse


def clean_label(label):
    # 仅去除标签中的多余标点符号，不再转换为小写
    return label.strip('.,\"\' ')


def process_data(input_file, matched_output_file, unmatched_output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(matched_output_file, 'w', encoding='utf-8') as matched_outfile, \
         open(unmatched_output_file, 'w', encoding='utf-8') as unmatched_outfile:
        for line in infile:
            item = json.loads(line)
            original_predicted_label = item.get('predicted_label', '')

            # four_class
            # patterns = [
            #     r'Label:\s*\**\s*([\w-]+)\s*\**\s*',
            #     r'Label:\s*\**\s*([\w-]+)[\s\n]*\**'
            #     r'as:\s*\**\s*([\w-]+)\s*\**',  # 匹配"as:"后的单词，现在包含连字符
            #     r'as\s+([\'\"]?)([\w-]+)\1',  # 新增：匹配 "as Non-ESG" 或 "as 'Non-ESG'" 或 "as \"Non-ESG\""
            #     r'is:\s*\**\s*([\w-]+)\s*\**',  # 匹配"is:"后的单词，现在包含连字符
            #     r'is\s+([\'\"]?)([\w-]+)\1',  # 新增：匹配 "is Non-ESG" 或 "is 'Non-ESG'" 或 "is \"Non-ESG\""
            #     r'Label:\s*\**\s*([^\n]+)\**',  # 匹配"Label:"后的任意非换行字符
            #     r'classify it as(?: label)?\s*:\s*\**\s*([^\n]+)\**',  # "classify it as label:" 后的任意非换行字符
            #     r'classify the text as\s*:\s*\**\s*([^\n]+)\**',  # "classify the text as:" 后的任意非换行字符
            #     r'label of\s*[\'\"]?([\w-]+)[\'\"]?',  # 新增：匹配 "label of Non-ESG" 或 "label of 'Non-ESG'"
            #     r'under the label\s*([\'\"]?)([\w-]+)\1',  # 新增：匹配 "under the label Non-ESG" 或 "under the label 'Non-ESG'"
            #     r'under the\s*([\'\"]?)([\w-]+)\1\s*label',  # 新增：匹配 "under the 'Non-ESG' label" 或 "under the Non-ESG label"
            #     r'\"(.*?)\"',  # 匹配双引号内的文本
            # ]

            # nine_class
            patterns = [
                r'Label:\s*([^\n]+?)(?=\s*(?:Explanation:|\n\n|$))',  # 匹配"Label:"后的文本，直到"Explanation:"、两个换行符或字符串结束
                r'under the\s*\"([^\"]+?)\"',  # 匹配"under the"后双引号内的内容，不限制引号后的字符
                r'under\s*\"([^\"]+?)\"',  # 匹配"under the"后双引号内的内容，不限制引号后的字符
                r'as\s*\"([^\"]+?)\"',  # 匹配"under the"后双引号内的内容，不限制引号后的字符
                r'under the label\s*\"([^\"]+?)\"',  # 匹配"under the"后双引号内的内容，不限制引号后的字符
                r'under the label of\s*\"([^\"]+?)\"',  # 匹配"under the label of"后双引号内的内容，不限制引号后的字符
                r'category of\s*\"([^\"]+?)\"',  # 匹配"under the label of"后双引号内的内容，不限制引号后的字符
                r'under\s+\"?([^\"]+?)\"?\s*(?=[.\n])',  # 断言后面是句号或换行符
                r'\"([^\"]+?)\"\s*category of\s*\"([^\"]+?)\.\"',  # 匹配引号内的文本，后跟"category of"和另一组引号内的文本
                r'category of\s+([\'\"]?)([\w\s-]+?)\1(?=[.\n])',  # 匹配 "category of Pollution and Waste." 或 "category of Pollution and Waste\n"
                r'label as:\s*\n*\n*\*?\s*([\'\"]?)([\w\s-]+?)\1\s*(?=[.\n])',  # 匹配 "label as:\n\n* Climate Change." 或 "label as:\n\nClimate Change\n"
                r'under\s+([\'\"]?)([\w\s-]+?)\1(?=[.\n])',  # 匹配 "under Corporate Governance." 或 "under Corporate Governance\n"
                r'as:\s*\n*\n*\*\s*([\w\s-]+?):(?=[.\n])',  # 匹配 "as:\n\n* Climate Change:" 结尾为点号或换行
                r'label of\s+([\'\"]?)([\w\s-]+?)\1(?=[.\n])'  # 匹配 "label of Pollution and Waste." 或 "label of Pollution and Waste\n"
            ]

            
            # 初始化一个空列表来收集所有匹配的结果
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, original_predicted_label, re.IGNORECASE | re.DOTALL | re.DOTALL))

            matched = False
            for match in matches:
                if match:  # 检查是否存在匹配项
                    cleaned_match = clean_label(''.join(match))  # 清洗第一个匹配的标签
                    item['response'] = original_predicted_label
                    item['predicted_label'] = cleaned_match
                    json.dump(item, matched_outfile, ensure_ascii=False)
                    matched_outfile.write('\n')
                    matched = True
                    break  # 找到第一个匹配项后结束循环

            if not matched:  # 如果没有找到匹配项
                item['response'] = original_predicted_label
                item['predicted_label'] = "Other"  # 未匹配标签设为"Other"
                json.dump(item, unmatched_outfile, ensure_ascii=False)
                unmatched_outfile.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model predictions.')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('matched_output_file', help='Output file path for matched labels')
    parser.add_argument('unmatched_output_file', help='Output file path for unmatched labels')

    args = parser.parse_args()
    process_data(args.input_file, args.matched_output_file, args.unmatched_output_file)
