import torch
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import load_dataset
import parse
import argparse
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import json

local_rank = int(os.environ.get('LOCAL_RANK', 0))
label_map = {
    "Env": 0,
    "Soc": 1,
    "Gov": 2,
    "Non-ESG": 3
}
# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning a Transformer model on a text classification task")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model identifier from Huggingface Transformers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


# 训练函数
def train(model, train_loader, optimizer, device, scaler):
    model.train()
    train_loss, n_correct, n_train = 0, 0, 0

    # 使用tqdm包装DataLoader以显示进度条
    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * labels.size(0)
        n_correct += (torch.argmax(outputs.logits, dim=1) == labels).sum().item()
        n_train += labels.size(0)

    avg_loss = train_loss / n_train
    accuracy = n_correct / n_train

    return avg_loss, accuracy


# 评估函数
def evaluate(model, val_loader, device):
    model.eval()
    eval_loss, n_correct, n_eval = 0, 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 在模型调用中提供labels参数，以便自动计算损失
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # 损失
            eval_loss += outputs.loss.item() * labels.size(0)
            
            # 计算准确率所需的值
            logits = outputs.logits
            n_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            n_eval += labels.size(0)

            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            true_labels.extend(labels.tolist())

    avg_loss = eval_loss / n_eval
    accuracy = n_correct / n_eval
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    acc = accuracy_score(true_labels, predictions)

    return avg_loss, precision, recall, f1, acc



class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 使用label_map将文本标签转换为整数
                self.data.append((item['text'], label_map[item['label']]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)  # 移除批次维度
        attention_mask = inputs['attention_mask'].squeeze(0)  # 移除批次维度
        # 将整数标签转换为张量
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_tensor}


def load_datasets(train_file, val_file, test_file):
    # 加载训练集
    tokenizer = BertTokenizer.from_pretrained('open-source-models/bert-base-uncased')

    train_dataset = TextDataset(train_file, tokenizer)
    # 加载验证集
    val_dataset = TextDataset(val_file, tokenizer)
    # 加载测试集
    test_dataset = TextDataset(test_file, tokenizer)
    return train_dataset, val_dataset, test_dataset


def main():
    args = parse_args()

    # 初始化分布式训练环境
    # torch.distributed.init_process_group(backend='nccl')
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.model_name = 'open-source-models/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=4).to(device)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scaler = GradScaler()
    # criterion = CrossEntropyLoss()

    # 直接从jsonl文件加载数据集
    train_dataset, val_dataset, test_dataset = load_datasets(
        'four_class_dataset/stratified/6_2_2/train.jsonl',
        'four_class_dataset/stratified/6_2_2/val.jsonl',
        'four_class_dataset/stratified/6_2_2/test.jsonl'
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        print(f"---------------Epoch {epoch+1}---------------")
        train_loss, accuracy = train(model, train_loader, optimizer, device, scaler)
        print(f"Train set: Train Loss:{train_loss}, Accuracy: {accuracy}")
        valid_loss, precision, recall, f1, acc = evaluate(model, val_loader, device)
        print(f"Valid set: Precision: {precision}, Recall: {recall}, F1: {f1}, Valid Loss: {valid_loss}, Accuracy: {acc}")

    # 测试集评估
    test_loss, precision, recall, f1, acc = evaluate(model, test_loader, device)
    print(f"Test: Precision: {precision}, Recall: {recall}, F1: {f1}, Test Loss:{test_loss}, Accuracy: {acc}")


if __name__ == "__main__":
    main()
