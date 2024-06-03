import json
import os


def extract_explanation(raw_explanation):
    # 查找"Explanation:"文本并提取之后的内容
    parts = raw_explanation.split("Explanation: ", 1)
    return parts[1].strip() if len(parts) > 1 else raw_explanation


def construct_and_save_sft_data_correctly(folder_path, output_file):
    categories = ['env', 'soc', 'gov']

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for category in categories:
            file_path = os.path.join(folder_path, category, 'correct_data.jsonl')
            
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    category_label = 'Env' if category == 'env' else 'Soc' if category == 'soc' else 'Gov'
                    label = data["label"]
                    
                    sft_data_items = [{
                        "instruction": "If the following text is ESG related data.",
                        "input": data["text"],
                        "output": data["explanation"],
                    }, {
                        "instruction": f"Classify the following text into one of the four ESG categories: 'Env', 'Soc', 'Gov', or 'Non-ESG'.",
                        "input": data["text"],
                        "output": f"Label: '{category_label}'. Explanation: '{extract_explanation(data['explanation2'])}'",
                    }, {
                        "instruction": "Classify the following text into one of the nine ESG categories: 'Climate Change', 'Natural Capital', 'Pollution and Waste', 'Human Capital', 'Product Liability', 'Community Relations', 'Corporate Governance', 'Business Ethics and Values', or 'Non-ESG'.",
                        "input": data["text"],
                        "output": f"Label: '{label}'. Explanation: '{extract_explanation(data['explanation3'])}'",
                    }]
                    
                    for item in sft_data_items:
                        text_with_input = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
                        item["text"] = text_with_input
                        
                        # Write the SFT data to file
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

# Specify the path to your 'output_9class' folder and output file
folder_path = '../output_9class'
output_file = 'esg_classification_sft.jsonl'

construct_and_save_sft_data_correctly(folder_path, output_file)

print("SFT data with corrected text field has been generated and saved.")
