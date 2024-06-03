from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 加载测试数据
test_data = []

with open('four_class_dataset/stratified/70_15_15/test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))

# with open('nine_class_dataset/stratified/81_09_10/test.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         test_data.append(json.loads(line))

# 加载模型和分词器

# four class
finbert = BertForSequenceClassification.from_pretrained('open-source-models/finbert-esg', num_labels=4)
tokenizer = BertTokenizer.from_pretrained('open-source-models/finbert-esg')

# nine class
# finbert = BertForSequenceClassification.from_pretrained('open-source-models/finbert-esg-9-categories', num_labels=9)
# tokenizer = BertTokenizer.from_pretrained('open-source-models/finbert-esg-9-categories')

# 创建pipeline
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer, truncation=True, max_length=512)

# four class
label_map = {"Environmental": "Env", "Social": "Soc", "Governance": "Gov", "None": "Non-ESG"}
reverse_label_map = {"Env": "Environmental", "Soc": "Social", "Gov": "Governance", "Non-ESG": "None"}

# nine class
# label_map = {
#     "Climate Change": "Climate Change",
#     "Natural Capital": "Natural Capital",
#     "Pollution & Waste": "Pollution and Waste",
#     "Human Capital": "Human Capital",
#     "Product Liability": "Product Liability",
#     "Community Relations": "Community Relations",
#     "Corporate Governance": "Corporate Governance",
#     "Business Ethics & Values": "Business Ethics and Values",
#     "Non-ESG": "Non-ESG"
# }
# reverse_label_map = {v: k for k, v in label_map.items()}

true_labels = []
predicted_labels = []
results = []

# 对每个测试实例进行预测
for item in test_data:
    text = item['text']
    true_label = item['label'] # when evalue nine class change label to sub_label
    result = nlp(text)[0]  # 获取预测结果
    predicted_label = label_map[result['label']]
    
    # 记录真实标签和预测标签
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    
    # 将预测结果保存到列表中
    results.append({'text': text, 'true_label': true_label, 'predicted_label': predicted_label})

# 保存预测结果到jsonl文件中
with open('prediction/four_class/finbert-esg/predicted_results.jsonl', 'w', encoding='utf-8') as f:
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
