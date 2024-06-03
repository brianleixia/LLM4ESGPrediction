from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 指定模型和分词器路径
# model_path = "model/nine_class/open-source-models/bert-base-uncased/best_model"
# model_path = "model/nine_class/output/bert/bert-epoch25/best_model"
# model_path = "model/nine_class/open-source-models/distilroberta-base/best_model"
# model_path = "model/nine_class/open-source-models/roberta-base/best_model"
# model_path = "model/nine_class/output/distilroberta/distilroberta-epoch25/best_model"
model_path = "model/nine_class/output/roberta/roberta-epoch25/best_model"

# 加载模型和分词器
# model = BertForSequenceClassification.from_pretrained(model_path, num_labels=9)
# tokenizer = BertTokenizer.from_pretrained(model_path)

model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=9)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# 定义标签映射
# four class
# label_map = {
#     "Env": 0,
#     "Soc": 1,
#     "Gov": 2,
#     "Non-ESG": 3
# }
# reverse_label_map = {v: k for k, v in label_map.items()}

# nine class
label_map = {
    "Climate Change": 0,
    "Natural Capital": 1,
    "Pollution and Waste": 2,
    "Human Capital": 3,
    "Product Liability": 4,
    "Community Relations": 5,
    "Corporate Governance": 6,
    "Business Ethics and Values": 7,
    "Non-ESG": 8
}
reverse_label_map = {v: k for k, v in label_map.items()}

# 加载测试数据
test_data = []
with open('nine_class_dataset/stratified/81_09_10/test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))

true_labels = []
predicted_labels = []
results = []

model.eval()  # 设置模型为评估模式

for item in test_data:
    text = item['text']
    true_label = item['sub_label'] # change 'label' to 'sub_label' when nine class
    
    # 分词和准备模型输入
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_label = reverse_label_map[predicted_class_id]
    
    # 记录标签
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    
    # 将预测结果保存到列表中
    results.append({'text': text, 'true_label': true_label, 'predicted_label': predicted_label})

# 保存预测结果到jsonl文件中
with open('prediction/nine_class/esg-roberta/predicted_results.jsonl', 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

# 计算每个类别的精确度、召回率、F1得分
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=list(reverse_label_map.values()), average=None)

# 计算整体的精确度
accuracy = accuracy_score(true_labels, predicted_labels)

# 输出整体的精确度、召回率、F1得分和准确度
print(f"Overall Precision: {precision.mean():.4f}, Recall: {recall.mean():.4f}, F1: {f1.mean():.4f}, Accuracy: {accuracy:.4f}")

# 输出每个类别的精确度、召回率、F1得分
for label_name, label_id in label_map.items():
    idx = label_id  # 直接使用label_id作为索引
    print(f"{label_name}: Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}, F1: {f1[idx]:.4f}")

