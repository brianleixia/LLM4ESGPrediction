from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

# 加载测试数据
test_data = []

with open('environmental_2k/test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))

# 加载模型和分词器

# two class
esgbert_env = AutoModelForSequenceClassification.from_pretrained('../open-source-models/EnvRoBERTa-environmental', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('../open-source-models/EnvRoBERTa-environmental')


# 创建pipeline
nlp = pipeline("text-classification", model=esgbert_env, tokenizer=tokenizer, truncation=True, max_length=512)

# four class
# label_map = {"Environmental": "Env", "Social": "Soc", "Governance": "Gov", "None": "Non-ESG"}
# reverse_label_map = {"Env": "Environmental", "Soc": "Social", "Gov": "Governance", "Non-ESG": "None"}

label_map = {"none": "0", "environmental": "1"}
reverse_label_map = {"0": "none", "1": "environmental"}

true_labels = []
predicted_labels = []
results = []

# 对每个测试实例进行预测
for item in test_data:
    text = item['text']
    true_label = item['env'] # when evalue nine class change label to env, soc, gov
    result = nlp(text)[0]  # 获取预测结果
    # print(result)
    predicted_label = label_map[result['label']]
    
    # 记录真实标签和预测标签
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    
    # 将预测结果保存到列表中
    results.append({'text': text, 'true_label': true_label, 'predicted_label': predicted_label})

# 保存预测结果到jsonl文件中
output_dir = 'prediction/envroberta/env'
os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 参数意味着如果目录已经存在，不会抛出异常

output_file_path = os.path.join(output_dir, 'predicted_results.jsonl')
with open(output_file_path, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

# 计算每个类别的精确度、召回率、F1得分
precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, labels=list(reverse_label_map.keys()), average=None)

# 计算整体的精确度
accuracy = accuracy_score(true_labels, predicted_labels)

# 输出整体的精确度、召回率、F1得分和准确度
print(f"Overall Precision: {precision.mean():.4f}, Recall: {recall.mean():.4f}, F1: {f1.mean():.4f}, Accuracy: {accuracy:.4f}")

# 输出每个类别的精确度、召回率、F1得分
for label, p, r, f in zip(reverse_label_map.keys(), precision, recall, f1):
    print(f"{label}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
