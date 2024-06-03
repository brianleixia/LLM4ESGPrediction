import json
import pandas as pd
import re
import os


def process_climate_fever(input_file_path, output_file_path):
    '''
        处理climate-fever-dataset, 提取里面的claim字段，以及evidences中的evidence
    '''

    with open(input_file_path, 'r') as file, open(output_file_path, 'w') as output_file:
        for line in file:
            data = json.loads(line) 
            claim = data.get("claim", "").strip()  # Trim spaces from claim
            evidences = data.get("evidences", [])

            # Write the trimmed claim
            output_file.write(f"{claim}\n")

            # Write each trimmed evidence
            for evidence in evidences:
                evidence_text = evidence.get("evidence", "").strip()  # Trim spaces from evidence
                output_file.write(f"{evidence_text}\n")


def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Modify the regex to keep certain special characters
    # Keep !, @, #, $, %, &, *, (, ), {, }, [, ], ;, :, ', ", <, >, ,, ., ?, /
    text = re.sub(r'[^\w\s!@#$%&*(){}\[\];:\'\",<>.?/]', '', text)

    return text.strip()


def process_esg_prospectus_clarify(input_csv_path, output_file_path):
    '''
        处理/data/ESG-Prospectus-Clarity-Category，下的文件内容，提取text部分
    '''
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Extract the 'Text' column
    texts = df['Text']

    # Clean and write each text to the output file
    with open(output_file_path, 'w') as output_file:
        for text in texts:
            cleaned_text = clean_text(str(text))
            output_file.write(f"{cleaned_text}\n")



def process_dax_esg_media(input_csv_path, output_file_path):
    '''
        处理data/DAX_ESG_Media，下的文件内容，提取content部分
    '''
    # Read the CSV file
    # df = pd.read_csv(input_csv_path)
    esg_documents_df = pd.read_csv(input_csv_path, sep="|")

    # Extract the 'Text' column
    texts = esg_documents_df['content']

    # Clean and write each text to the output file
    with open(output_file_path, 'w') as output_file:
        for text in texts:
            cleaned_text = clean_text(str(text))
            output_file.write(f"{cleaned_text}\n")


def process_esgbert_base(input_csv_path, output_file_path):
    '''
        处理/workspace/dissertation/data/ESGBERT_base_data，下的文件内容
    '''
    # Read the CSV file
    # df = pd.read_csv(input_csv_path)
    # df = pd.read_csv(input_csv_path, header=None, names=['content'])
    df = pd.read_csv(input_csv_path)

    # Extract the 'Text' column
    texts = df['sentence']

    # Clean and write each text to the output file
    with open(output_file_path, 'w') as output_file:
        for text in texts:
            cleaned_text = clean_text(str(text))
            output_file.write(f"{cleaned_text}\n")


def process_environmental_claims(input_directory, output_file_path):
    '''
        处理data/environmental_claims/data中的 .parquet 文件，并将 'text' 列的内容保存到一个输出文件中。

        :param input_directory: 输入文件夹的路径，包含 .parquet 文件
        :param output_file_path: 输出文件的路径
    '''

    # 打开输出文件
    with open(output_file_path, 'w') as output_file:

        # 遍历文件夹中的所有 .parquet 文件
        for filename in os.listdir(input_directory):
            if filename.endswith(".parquet"):
                file_path = os.path.join(input_directory, filename)

                # 读取 .parquet 文件
                df = pd.read_parquet(file_path)

                # 检查 'text' 列是否存在
                if 'text' in df.columns:
                    # 遍历 'text' 列并写入文件
                    for text in df['text']:
                        cleaned_text = clean_text(str(text))
                        output_file.write(f"{cleaned_text}\n")
                else:
                    print(f"'text' column not found in {filename}")


def process_esg_sentiment(input_directory, output_file_path):
    '''
    处理data/esg-sentiment/data中的 .parquet 文件，并将 'text' 列的内容保存到一个输出文件中。

    :param input_directory: 输入文件夹的路径，包含 .parquet 文件
    :param output_file_path: 输出文件的路径
    '''

    # 打开输出文件
    with open(output_file_path, 'w') as output_file:

        # 遍历文件夹中的所有 .parquet 文件
        for filename in os.listdir(input_directory):
            if filename.endswith(".parquet"):
                file_path = os.path.join(input_directory, filename)

                # 读取 .parquet 文件
                df = pd.read_parquet(file_path)

                # 检查 'Text' 列是否存在
                if 'Text' in df.columns:
                    # 遍历 'Text' 列并写入文件
                    for text in df['Text']:
                        cleaned_text = clean_text(str(text))
                        output_file.write(f"{cleaned_text}\n")
                else:
                    print(f"'Text' column not found in {filename}")



def split_text_by_label(input_csv_path, output_path_0, output_path_1):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv_path)

    # 根据标签分割数据
    texts_0 = df[df['gov'] == 0]['text']
    texts_1 = df[df['gov'] == 1]['text']

    # 将每个分组的文本写入不同的文件
    with open(output_path_0, 'w') as file_0, open(output_path_1, 'w') as file_1:
        for text in texts_0:
            cleaned_text = text.replace('\n', ' ')  # 替换换行符为一个空格
            file_0.write(cleaned_text + '\n')
        for text in texts_1:
            cleaned_text = text.replace('\n', ' ')  # 替换换行符为一个空格
            file_1.write(cleaned_text + '\n')



if __name__ == "__main__":
    # process climate-fever-dataset
    climate_fever_in_path = 'data/climate-fever-dataset/climate-fever-dataset-r1.jsonl'  
    climate_fever_out_path = 'processed_data/extract_text/climate-fever-dataset.txt'  
    # process_climate_fever(climate_fever_in_path, climate_fever_out_path)


    # process esg_prospectus_clarify
    esg_prospectus_in_path = 'data/ESG-Prospectus-Clarity-Category/esg-prospectus-clarity-category.csv'  # Replace with your actual file path
    esg_prospectus_out_path = 'processed_data/extract_text/esg-prospectus-clarity-category.txt'  # Replace with your desired output file path
    # process_esg_prospectus_clarify(esg_prospectus_in_path, esg_prospectus_out_path)


    # process dax_esg_media
    dax_esg_media_in_path = 'data/DAX_ESG_Media/esg_documents_for_dax_companies.csv'  # Replace with your actual file path
    dax_esg_media_out_path = 'processed_data/extract_text/dax_esg_media.txt'  # Replace with your desired output file path
    # process_dax_esg_media(dax_esg_media_in_path, dax_esg_media_out_path)


    # process esgbert_base
    esgbert_base_in_path = 'data/ESGBERT_base_data/base_data.csv'  # Replace with your actual file path
    esgbert_base_out_path = 'processed_data/extract_text/esgbert_base_data.txt'  # Replace with your desired output file path
    # process_esgbert_base(esgbert_base_in_path, esgbert_base_out_path)


    # process environmental_claims
    environmental_claims_in_path = "data/environmental_claims/data"
    environmental_claims_out_path = "processed_data/extract_text/environmental_claims.txt"
    # process_environmental_claims(environmental_claims_in_path, environmental_claims_out_path)


    # process 
    environmental_claims_in_path = "data/esg-sentiment/data"
    environmental_claims_out_path = "processed_data/extract_text/esg_sentiment.txt"
    process_esg_sentiment(environmental_claims_in_path, environmental_claims_out_path)


    
