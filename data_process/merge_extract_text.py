import os
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

'''
    将提取出的text全部合并到一起
'''

def merge_and_tokenize_txt_files_with_progress(directory, output_file_path):
    # Ensure that the NLTK punkt tokenizer is available
    nltk.download('punkt')

    # List all .txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    # Open the output file
    with open(output_file_path, 'w') as output_file:
        # Process each file
        for txt_file in txt_files:
            print(f"Processing file: {txt_file}")
            with open(os.path.join(directory, txt_file), 'r') as file:
                # Read and process the content of the file line by line
                for line in tqdm(file, desc=f"Processing {txt_file}"):
                    # Tokenize the content of the line into sentences
                    sentences = sent_tokenize(line)

                    # Write each sentence to the output file on a new line
                    for sentence in sentences:
                        output_file.write(sentence + "\n")
                        output_file.flush()

# Example usage
directory = 'processed_data/extract_text'  
output_file_path = 'merged_output.txt'  
merge_and_tokenize_txt_files_with_progress(directory, output_file_path)  # Run the function
