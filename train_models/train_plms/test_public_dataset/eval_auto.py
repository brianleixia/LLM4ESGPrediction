import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

def main(model_path, output_dir, data_type):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    test_data = []
    if data_type == 'env':
        folder_prefix = "environmental"
    elif data_type == 'soc':
        folder_prefix = "social"
    elif data_type == 'gov':
        folder_prefix = "governance"

    with open(f'{folder_prefix}_2k/test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    true_labels = []
    predicted_labels = []
    results = []

    model.eval()  # Set model to evaluation mode

    for item in test_data:
        text = item['text']
        true_label = item[data_type]  # change env, soc, or gov based on input argument
        
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = str(predicted_class_id)
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        
        results.append({'text': text, 'true_label': true_label, 'predicted_label': predicted_label})

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'predicted_results.jsonl')

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Overall Precision: {precision.mean():.4f}, Recall: {recall.mean():.4f}, F1: {f1.mean():.4f}, Accuracy: {accuracy:.4f}")

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ESG prediction model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pretrained model directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save prediction results")
    parser.add_argument('--data_type', type=str, required=True, choices=['env', 'soc', 'gov'], help="Type of data to predict on: env, soc, or gov")
    
    args = parser.parse_args()
    
    main(args.model_path, args.output_dir, args.data_type)
