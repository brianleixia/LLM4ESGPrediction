import random
from http import HTTPStatus
import dashscope
import json
import argparse
import openai
import backoff
import os
from tqdm import tqdm
import re


model_name="gpt-3.5-turbo"


@backoff.on_exception(
    backoff.expo,
    (openai.error.OpenAIError, openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError,
     openai.error.APIConnectionError),
    max_tries=5
)
def call_with_messages(text, esg_type, api_key, temperature=0.5):
    env_messages = [
    {'role': 'system', 'content': 'You are an expert in classifying environmental ESG data into finer categories such as Climate Change, Natural Capital, and Pollution and Waste.'},
    {'role': 'user', 'content': f"For Environmental ESG data classification, consider: \n\n\
        Climate Change: Focuses on actions and policies related to reducing carbon emissions, enhancing carbon efficiency, and adopting \
            renewable energy sources. It includes the eco-innovative design and retrofitting of buildings to minimize environmental impact. \n\
        Natural Capital: Encompasses sustainable water management, biodiversity conservation, responsible land use, and eco-friendly sourcing \
            of raw materials like seafood, timber, palm oil, beef and cotton, aiming to reduce environmental degradation. \n\
        Pollution and Waste: Covers management and reduction of toxic emissions, sustainable packaging practices, and the disposal or recycling \
            of electronic waste, aiming to minimize pollution and waste impact on the environment. \n\n\
        First assign a class label based on these categories, or 'other' if uncertain. Then give a explanation \n\n\
        Example 1: 'Beginning in 2012 through the end of 2016, we have converted 19 plants from coal to natural gas or steam.' \n\
        Label: Climate Change. \n\
        Example 2: '100% of the paper used in our U.S direct marketing efforts was certified to be from sustainably managed forests.' \n\
        Label: Natural Capital. \n\
        Example 3: 'All of these programs reduce waste and encourage reuse by ensuring that valuable products can go back into the hands of customers rather than being sent to landfills.' \n\
        Label: Pollution and Waste. \n\n\
        Text: {text} \n\
        Label: ***class-label***. Let's think step-by-step."}
    ]

    soc_messages = [
    {'role': 'system', 'content': 'You are an expert in classifying social ESG data into finer categories such as Human Capital, Product Liability, and Community Relations.'},
    {'role': 'user', 'content': f"For social ESG data classification, consider: \n\n\
        Human Capital: Involves labor management, employee health and safety, training and development, \
            and ethical supply chain practices. Key aspects include workforce diversity, pay equality, safety policies, \
            employee engagement, and standards against exploitative labor practices in the supply chain. \n\
        Product Liability: Relates to product safety, quality, data privacy, chemical safety, and financial transparency. \
            It encompasses issues like product recalls, data breaches, ethical use of chemicals, consumer rights to financial \
            product transparency, and adapting to health and demographic trends. \n\
        Community Relations: Focuses on a company's engagement with local communities, including improving access to essential \
            services like healthcare, finance, and communication. It covers efforts to serve historically underserved markets \
            and philanthropic initiatives aimed at community development. \n\n\
        First assign a class label based on these categories, or 'other' if uncertain. Then give a explanation \n\n\
        Example 1: 'Enterprise Leadership delivers targeted leadership development programs to colleagues at specific stages of their careers.' \n\
        Label: Human Capital. \n\
        Example 2: 'Our products are designed and tested to comply with all applicable safety regulations in the countries where the products are sold.' \n\
        Label: Product Liability. \n\
        Example 3: 'We aim to improve access for local people to health care and treatments for diseases such as cancer' \n\
        Label: Community Relations. \n\n\
        Text: {text} \n\
        Label: ***class-label***. Let's think step-by-step."}
    ]
    
    gov_messages = [
    {'role': 'system', 'content': 'You are an expert in classifying governance ESG data into finer categories such as Corporate Governance, and Business Ethics and Values.'},
    {'role': 'user', 'content': f"For governance ESG data classification, consider: \n\n\
        Corporate Governance: Encompasses practices and policies related to shareholder relations, board composition, executive compensation, \
            and internal controls. Key aspects include ownership structure, board independence and diversity, transparency in executive pay aligned \
            with performance, and robust internal auditing processes. \n\
        Business Ethics and Values: Covers the ethical conduct and integrity in business operations, including anti-corruption, compliance, \
            transparency, and accountability. It addresses issues such as fraud prevention, fair dealing, political contributions, and how a company responds \
            to ethical dilemmas and controversies. \n\n\
        First assign a class label based on these categories, or 'other' if uncertain. Then give a explanation \n\n\
        Example 1: 'Our Board is composed entirely of independent directors other than our chairman and CEO, \
            and is diverse, with diversity reflecting gender, age, race, ethnicity, background, professional \
            experience, and perspectives.' \n\
        Label: Corporate Governance. \n\
        Example 2: 'Huntington is dedicated to uncompromising integrity in all that it does and how it relates to its nternal colleagues and to persons outside Huntington.' \n\
        Label: Business Ethics and Values. \n\n\
        Text: {text} \n\
        Label: ***class-label***. Let's think step-by-step."}
    ]

    # print(env_messages)
    if esg_type == 'env':
        messages = env_messages
    elif esg_type == 'soc':
        messages = soc_messages
    elif esg_type == 'gov':
        messages = gov_messages

    try: 
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            api_key=api_key,
        )

        usage = response['usage']

        label, explanation = determine_environmental_from_response(response)  # Implement this function based on your response format

        return label, explanation, usage
    
    except openai.error.RateLimitError as e:
        raise Exception(f"Error: {e.user_message}")
    except openai.error.ServiceUnavailableError as e:
        print("Server is currently unavailable. Waiting for a longer time before retrying...")
        time.sleep(60)  # wait for 60 seconds before retrying
        raise e  # re-raise the exception so that backoff can handle the retry
    except openai.error.OpenAIError as e:
        print(f"OpenAI Error: {e}")
        print(f"Error occurred with data: {text}")
        return 'No content available', '', {}  # Return default value after all retries are exhausted
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Error occurred with data: {text}")
        return 'No content available', '', {}  # Return default value for any other unexpected error


def determine_environmental_from_response(response):
    # 获取响应内容
    explanation = response['choices'][0]['message']['content']

    # 定义正则表达式以非贪婪方式匹配标签，直到遇到第一个换行符或字符串末尾
    label_pattern = r'Label: ([^\n\.]+)'

    # 尝试匹配标签
    label_match = re.search(label_pattern, explanation)

    # 提取标签，如果没有找到匹配，则标签为"Unknown"
    label = label_match.group(1).strip() if label_match else "Unknown"

    return label, explanation


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


def process_jsonl_file(data, all_results_path, usage_file, esg_type, api_key):
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
                label, explanation, usage = call_with_messages(item['text'], esg_type, api_key)
                save_usage_data(usage, usage_file)

                item.update({'label': label, 'explanation3': explanation})
                # print(item)
                save_result_as_jsonl(item, all_results_path)

            pbar.update(1)  



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select high-quality data from text file.")
    parser.add_argument('--data_file', type=str, help='Path to the input data file')
    parser.add_argument('--all_results_path', type=str, help='Path to save all results')
    parser.add_argument('--usage_file', type=str, help="The output file where usage will be stored")
    parser.add_argument('--esg_type', type=str, help="Which typy needs to process: env, soc, gov")

    args = parser.parse_args()

    all_results_path = args.all_results_path
    usage_file = args.usage_file
    esg_type = args.esg_type

    os.environ["OPENAI_API_KEY"] = "rWFLxW6GXvo2c9smrGKlIFwRVQsixwOl"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = "https://gptproxy.llmpaas.woa.com/v1"  # 只增加这一行即可

    data = load_data(args.data_file)
    process_jsonl_file(data, all_results_path, usage_file, esg_type, os.environ["OPENAI_API_KEY"])
