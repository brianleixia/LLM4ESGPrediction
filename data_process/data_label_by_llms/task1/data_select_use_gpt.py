import random
from http import HTTPStatus
import dashscope
import json
import argparse
import openai
import backoff
import os

model_name="gpt-3.5-turbo"

@backoff.on_exception(
    backoff.expo,
    (openai.error.OpenAIError, openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError,
     openai.error.APIConnectionError),
    max_tries=5
)
def call_with_messages(data, api_key, temperature=0.5):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant in data managing, and good at using high-quality data criteria for ESG content selection.'},
        {'role': 'user', 'content': f"To identify high-quality ESG data, we should consider the following criteria: \n, \
                1) Relevance: The content should be related to environmental, social, or governance issues or relevant topics.\n \
                2) Accuracy: The information should be factually correct and up-to-date.\n \
                3) Source Credibility: Information should come from reputable sources, such as established news outlets, government reports, or recognized experts in the field.\n \
                4) Specificity: The data should provide detailed insights or examples, rather than general statements.\n \
                5) Objectivity: The content should be free from bias and present a balanced view. \n\n \
                The following sentence is the data need to define \n: '{data}'. \n\n Evaluate it using the high-quality data criteria for ESG content. Don't be too strict. \n\
                Answer 'Yes' or 'No' first, then give the explanation. Let's think step by step"}
    ]
    # print(messages)

    try: 
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            api_key=api_key,
        )

        usage = response['usage']

        is_high_quality, explanation = determine_quality_from_response(response)  # Implement this function based on your response format
        
        result = { 
            "text": data,
            "is_high_quality": is_high_quality,
            "explanation": explanation
        }

        return result, usage
    
    except openai.error.RateLimitError as e:
        raise Exception(f"Error: {e.user_message}")
    except openai.error.ServiceUnavailableError as e:
        print("Server is currently unavailable. Waiting for a longer time before retrying...")
        time.sleep(60)  # wait for 60 seconds before retrying
        raise e  # re-raise the exception so that backoff can handle the retry
    except openai.error.OpenAIError as e:
        print(f"OpenAI Error: {e}")
        return 'No content available', {}  # Return default value after all retries are exhausted
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 'No content available', {}  # Return default value for any other unexpected error


def load_data(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


def determine_quality_from_response(response):
    """
    Determines if the data is high quality based on the model's response.
    Expects the response to start with 'Yes' or 'No', followed by an explanation.
    """
    try:
        # Extracting the assistant's response
        assistant_response = response['choices'][0]['message']['content'].strip()

        # Check if the response starts with 'Yes' or 'No'
        if assistant_response.startswith("Yes"):
            return True, assistant_response
        elif assistant_response.startswith("No"):
            return False, assistant_response
        else:
            print("Error: Response does not start with 'Yes' or 'No'.")
            return False, assistant_response
    except KeyError as e:
        print(f"KeyError encountered in response parsing: {e}")
        return False, "Error in parsing response"


def save_data(data, file_path):
    with open(file_path, 'w') as file:
        file.writelines(data)


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


def select_high_quality_data(data, target_count, all_results_path, high_quality_results_path, usage_file, api_key):
    # 加载已存在的高质量结果
    high_quality_results = load_existing_results(high_quality_results_path)
    if len(high_quality_results) >= target_count:
        print("Collect high quality data task done!")
        return high_quality_results  # 如果已经满足目标数量，直接退出

    # 这里缺少一个判断，需要讲已经有的结果从总数据里取出，避免随机取样取到
    all_results = load_existing_results(all_results_path)

    # 从 data 中移除已经处理过的结果
    processed_texts = {result["text"] for result in all_results}
    data = [d for d in data if d not in processed_texts]

    processed_texts_size = len(processed_texts)
    data_size = len(data)

    print(f"****** {processed_texts_size} has processed, {data_size} left ******")

    while len(high_quality_results) < target_count and len(all_results) < data_size:
        # 随机抽取一小部分数据进行评估
        sample_size = min(500, target_count - len(high_quality_results), data_size - len(all_results))
        data_sampled = random.sample(data, sample_size)

        for d in data_sampled:
            result, usage = call_with_messages(d.strip(), api_key)
            save_usage_data(usage, usage_file)

            # 立即保存所有结果
            save_result_as_jsonl(result, all_results_path)
            all_results.append(result)  # 更新 all_results

            # 这里缺少一个异常判断
            if result.get("is_high_quality"):
                high_quality_results.append(result)

                # 立即保存高质量结果
                save_result_as_jsonl(result, high_quality_results_path)

                # 当 len(high_quality_results) 是500的倍数时打印
                if len(high_quality_results) % 500 == 0:
                    print(f"Current high quality results count: {len(high_quality_results)}")


            # 从原始数据集中移除已经评估过的数据
            data.remove(d)

            if len(high_quality_results) >= target_count:
                print("Collect high quality data task done!")
                break

    return high_quality_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select high-quality data from text file.")
    parser.add_argument('--data_file', type=str, help='Path to the input data file')
    parser.add_argument('--target_count', type=int, help='Number of high quality data points to select')
    parser.add_argument('--all_results_path', type=str, help='Path to save all results')
    parser.add_argument('--high_quality_results_path', type=str, help='Path to save high quality results')
    parser.add_argument('--usage_file', type=str, help="The output file where usage will be stored")

    args = parser.parse_args()
    target_count = args.target_count
    all_results_path = args.all_results_path
    high_quality_results_path = args.high_quality_results_path
    usage_file = args.usage_file

    os.environ["OPENAI_API_KEY"] = "rWFLxW6GXvo2c9smrGKlIFwRVQsixwOl"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = "https://gptproxy.llmpaas.woa.com/v1"  # 只增加这一行即可

    data = load_data(args.data_file)
    select_high_quality_data(data, target_count, all_results_path, high_quality_results_path, usage_file, os.environ["OPENAI_API_KEY"])
