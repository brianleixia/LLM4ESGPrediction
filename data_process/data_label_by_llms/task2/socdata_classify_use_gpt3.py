import random
from http import HTTPStatus
import dashscope
import json
import argparse
import openai
import backoff
import os
from tqdm import tqdm


model_name="gpt-3.5-turbo"

@backoff.on_exception(
    backoff.expo,
    (openai.error.OpenAIError, openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError,
     openai.error.APIConnectionError),
    max_tries=5
)
def call_with_messages(text, api_key, temperature=0.5):
    messages = [
        {'role': 'system', 'content': 'You are an expert in ESG data classification, especially social ESG data classification'},
        {'role': 'user', 'content': f"To identify social data of ESG data, consider the following criteria: \n, \
                Social criteria look at the company’s business relationships. Topics of interest are the support of diversity and human rights, \
                consumer protection, coherence of company values and those of the suppliers, donations or voluntary work in local communities, \
                issues of employee safety and health, and interests of other stakeholders.\n \
                Answer 'Yes' or 'No' first, then give the explanation. \n\n \
                Text: We published pay gap data inclusive of ethnicity for 2020 and 2021 with the gender gap for 2021 recorded as 29.3% and the ethnicity gap recorded as 5.7%. \n \
                Answer: Yes. \n \
                Text: Deutsche Beteiligungs AG employed 33 female and 28 male staff, making a total of 61 employees. \n \
                Answer: Yes, edge case. \n \
                Text: {text} \n \
                Answer: Let's think step by step."}
    ]

    try: 
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            api_key=api_key,
        )

        usage = response['usage']

        is_soc, explanation = determine_environmental_from_response(response)  # Implement this function based on your response format

        return is_soc, explanation, usage
    
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
        explanation = response['choices'][0]['message']['content']
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


def process_jsonl_file(data, all_results_path, usage_file, api_key):
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
                is_soc, explanation, usage = call_with_messages(item['text'], api_key)
                save_usage_data(usage, usage_file)

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
    usage_file = args.usage_file

    os.environ["OPENAI_API_KEY"] = "rWFLxW6GXvo2c9smrGKlIFwRVQsixwOl"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = "https://gptproxy.llmpaas.woa.com/v1"  # 只增加这一行即可

    data = load_data(args.data_file)
    process_jsonl_file(data, all_results_path, usage_file, os.environ["OPENAI_API_KEY"])
