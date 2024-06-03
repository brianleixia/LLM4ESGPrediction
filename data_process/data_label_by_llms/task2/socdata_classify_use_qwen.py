import random
from http import HTTPStatus
import dashscope
import json
import argparse
import openai
import backoff
import os
from tqdm import tqdm


# model = 'qwen-max-longcontext'
model = 'qwen-max-1201'


def call_with_messages(data, max_retries=5):
    messages = [
        {'role': 'system', 'content': 'You are a expert in ESG data classification, especially social ESG data classification'},
        {'role': 'user', 'content': f"To identify social data of ESG data, consider the following criteria: \n, \
                Social criteria look at the company’s business relationships. Topics of interest are the support of diversity and human rights, \
                consumer protection, coherence of company values and those of the suppliers, donations or voluntary work in local communities, \
                issues of employee safety and health, and interests of other stakeholders.\n \
                Answer 'Yes' or 'No' first, then give the explanation. \n\n \
                Text: We published pay gap data inclusive of ethnicity for 2020 and 2021 with the gender gap for 2021 recorded as 29.3% and the ethnicity gap recorded as 5.7%. \n \
                Answer: Yes. \n \
                Text: Deutsche Beteiligungs AG employed 33 female and 28 male staff, making a total of 61 employees. \n \
                Answer: Yes, edge case. \n \
                Text: {data} \n \
                Answer: Let's think step by step."}
    ]

    # print(messages)
    for attempt in range(max_retries):
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message',
        )
        if response.status_code == HTTPStatus.OK:
            is_soc, explanation = determine_environmental_from_response(response)
            return is_soc, explanation
        else:
            # print('Error: %s' % response.message)
            print(f'Error on attempt {attempt + 1}: {response.message}')
            # return False

    print(f"This text: '{data}', doesn't get result!")
    return 'No content available', '',


def determine_environmental_from_response(response):
        explanation = response['output']['choices'][0]['message']['content']
        is_soc = explanation.lower().startswith("yes") or "answer: yes" in explanation.lower()
        return is_soc, explanation


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON format in line: {line}")
    return data


def save_result_as_jsonl(result, file_path):
    with open(file_path, 'a') as file:  # 使用追加模式 'a'
        json_line = json.dumps(result)
        file.write(json_line + '\n')
        file.flush()


def save_usage_data(usage_data, usage_file):
    # Construct file path
    file_path = os.path.join(usage_file)

    # Ensure 'usage' directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Append the new usage_data as a new line to the file
    with open(file_path, 'a') as f:
        f.write(json.dumps(usage_data) + '\n')
        f.flush()  # Ensure data is written to the file


def load_existing_results(file_path):
    results = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                results.append(json.loads(line))
    except FileNotFoundError:
        pass  # 如果文件不存在，则继续执行
    return results


def process_jsonl_file(data, all_results_path):
    existing_results = load_existing_results(all_results_path)
    existing_texts = {result['text'] for result in existing_results}

    # updated_data = existing_results  # 初始化已有结果
    total_items = len(data)  # 数据总数
    processed_count = len(existing_results)  # 已处理数据计数

    with tqdm(total=total_items, desc="Processing", unit="item") as pbar:
        pbar.update(processed_count)  # 初始进度条更新到已处理的数量
        for item in data:
            # print(item['text'])
            if item['text'] not in existing_texts:
                is_soc, explanation = call_with_messages(item['text'])

                item.update({'is_soc': is_soc, 'explanation2': explanation})
                # print(item)
                save_result_as_jsonl(item, all_results_path)

            pbar.update(1)  



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select high-quality data from text file.")
    parser.add_argument('--data_file', type=str, help='Path to the input data file')
    parser.add_argument('--all_results_path', type=str, help='Path to save all results')
    parser.add_argument('--usage_file', type=str, help="The output file where usage will be stored")

    args = parser.parse_args()
    all_results_path = args.all_results_path


    data = load_data(args.data_file)
    process_jsonl_file(data, all_results_path)
