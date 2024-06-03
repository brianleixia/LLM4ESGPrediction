import json
import os
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import csv

def evaluate_predictions(parent_folder, csv_file):
    true_labels = []
    predicted_labels = []
    correct_predictions = []
    incorrect_predictions = []

    input_file_path = os.path.join(parent_folder, 'predicted_results.jsonl')
    if os.path.exists(input_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                true_labels.append(result['true_label'])
                predicted_labels.append(result['predicted_label'])

                if result['true_label'] == result['predicted_label']:
                    correct_predictions.append(result)
                else:
                    incorrect_predictions.append(result)

        correct_predictions_path = os.path.join(parent_folder, 'correct_predictions.jsonl')
        incorrect_predictions_path = os.path.join(parent_folder, 'incorrect_predictions.jsonl')

        with open(correct_predictions_path, 'w', encoding='utf-8') as f:
            for prediction in correct_predictions:
                f.write(json.dumps(prediction) + '\n')

        with open(incorrect_predictions_path, 'w', encoding='utf-8') as f:
            for prediction in incorrect_predictions:
                f.write(json.dumps(prediction) + '\n')

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=sorted(set(true_labels)), average=None)
        accuracy = accuracy_score(true_labels, predicted_labels)

        print(f"\nEvaluating: {parent_folder}")
        print(f"Precision: {precision.mean():.4f}, Recall: {recall.mean():.4f}, F1: {f1.mean():.4f}, Accuracy: {accuracy:.4f}")
        for label, p, r, f in zip(sorted(set(true_labels)), precision, recall, f1):
            print(f"{label}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

         # 将结果写入CSV文件
        with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([parent_folder, precision.mean(), recall.mean(), f1.mean(), accuracy])
            for label, p, r, f in zip(sorted(set(true_labels)), precision, recall, f1):
                csvwriter.writerow([label, p, r, f])

        
    else:
        print(f"\nFile not found: {input_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model predictions.')
    parser.add_argument('folders', nargs='+', help='List of folders containing prediction results to evaluate')
    parser.add_argument('--csv', default='evaluation_results.csv', help='CSV file to store evaluation results')
    
    args = parser.parse_args()

    # 检查CSV文件是否存在，如果不存在，则创建并添加表头
    if not os.path.exists(args.csv):
        with open(args.csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Folder', 'Precision', 'Recall', 'F1', 'Accuracy', 'Label', 'Label Precision', 'Label Recall', 'Label F1'])

    for parent_folder in args.folders:
        evaluate_predictions(parent_folder, args.csv)
