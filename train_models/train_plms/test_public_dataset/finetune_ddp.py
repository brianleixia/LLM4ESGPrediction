import torch
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import BertConfig
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import load_dataset
import argparse
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import os
import logging
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import json

local_rank = int(os.environ.get('LOCAL_RANK', 0))

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning a Transformer model on a text classification task")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased", help="Model Path")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model identifier from Huggingface Transformers")
    parser.add_argument("--model_type", type=str, default="bert", help="Model Type [bert, robert]")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimizer")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


def setup_logging(model_name, log_file='training.log'):
    log_dir = os.path.join('model', model_name, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


# 训练函数
def train(model, train_loader, optimizer, device, scaler, local_rank):
    model.train()
    train_loss, n_correct, n_train = 0, 0, 0

    # 使用tqdm包装DataLoader以显示进度条，仅在 local_rank 为 0 的进程中显示
    loader = tqdm(train_loader, desc="Training", leave=False) if local_rank == 0 else train_loader

    for batch in loader:
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

        reduced_loss = reduce_mean(loss, dist.get_world_size())
        train_loss += reduced_loss.item() * labels.size(0)
        n_correct += (torch.argmax(outputs.logits, dim=1) == labels).sum().item()
        n_train += labels.size(0)

    avg_loss = train_loss / n_train
    n_correct = torch.tensor(n_correct).to(device)
    n_train = torch.tensor(n_train).to(device)

    dist.all_reduce(n_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_train, op=dist.ReduceOp.SUM)

    accuracy = (n_correct / n_train).item() if n_train > 0 else 0.0

    return avg_loss, accuracy



# 评估函数
def evaluate(model, val_loader, device, local_rank):
    model.eval()
    eval_loss, n_correct, n_eval = 0, 0, 0
    # predictions, true_labels = [], []

    # 使用tqdm包装DataLoader以显示进度条，仅在 local_rank 为 0 的进程中显示
    loader = tqdm(val_loader, desc="Evaluating", leave=False) if local_rank == 0 else val_loader

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 在模型调用中提供labels参数，以便自动计算损失
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            reduced_loss = reduce_mean(loss, dist.get_world_size())
            eval_loss += reduced_loss.item() * labels.size(0)
            n_correct += (torch.argmax(outputs.logits, dim=1) == labels).sum().item()
            n_eval += labels.size(0)

    avg_loss = eval_loss / n_eval
    # accuracy = n_correct / n_train

    n_correct = torch.tensor(n_correct).to(device)
    n_eval = torch.tensor(n_eval).to(device)

    dist.all_reduce(n_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_eval, op=dist.ReduceOp.SUM)

    accuracy = (n_correct / n_eval).item() if n_eval > 0 else 0.0

    return avg_loss, accuracy


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 使用label_map将文本标签转换为整数
                # self.data.append((item['text'], label_map[item['label']]))
                # self.data.append((item['text'], item['env']))
                self.data.append((item['text'], item['soc']))
                # self.data.append((item['text'], item['gov']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)  # 移除批次维度
        attention_mask = inputs['attention_mask'].squeeze(0)  # 移除批次维度
        # 将整数标签转换为张量
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_tensor}


def load_datasets(tokenizer, train_file, val_file, test_file):
    # args = parse_args()
    # tokenizer = BertTokenizer.from_pretrained(args.model_name)
    # tokenizer = RobertaTokenizer.from_pretrained('output/distilroberta/distilroberta-epoch25')

    # model_name = args.model_name
    # model_type = args.model_type
    # if model_type == 'bert':
    #     tokenizer = BertTokenizer.from_pretrained(model_name)
    # elif model_type =='roberta':
    #     tokenizer = RobertaTokenizer.from_pretrained(model_name)

    train_dataset = TextDataset(train_file, tokenizer)
    # 加载验证集
    val_dataset = TextDataset(val_file, tokenizer)
    # 加载测试集
    test_dataset = TextDataset(test_file, tokenizer)
    return train_dataset, val_dataset, test_dataset


def main():
    args = parse_args()

    # 初始化分布式训练环境
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    model_name = args.model_name
    setup_logging(model_name)

    # tokenizer = BertTokenizer.from_pretrained(args.model_name)
    if args.model_type == 'bert':
        # config = BertConfig.from_pretrained(model_name, num_labels=4, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
        # model = BertForSequenceClassification.from_pretrained(model_name, config=config).to(device)
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif args.model_type =='roberta':
        model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=3, verbose=True)
    # criterion = CrossEntropyLoss()

    # 直接从jsonl文件加载数据集
    # train_dataset, val_dataset, test_dataset = load_datasets(
    #     tokenizer,
    #     'environmental_2k/train.jsonl',
    #     'environmental_2k/val.jsonl',
    #     'environmental_2k/test.jsonl'
    # )

    train_dataset, val_dataset, test_dataset = load_datasets(
        tokenizer,
        'social_2k/train.jsonl',
        'social_2k/val.jsonl',
        'social_2k/test.jsonl'
    )

    # train_dataset, val_dataset, test_dataset = load_datasets(
    #     tokenizer,
    #     'governance_2k/train.jsonl',
    #     'governance_2k/val.jsonl',
    #     'governance_2k/test.jsonl'
    # )

    # 创建DistributedSampler实例
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)

    # 使用DistributedSampler创建DataLoader
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              sampler=train_sampler, 
                              shuffle=False, 
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            sampler=val_sampler, 
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.batch_size, 
                             sampler=test_sampler, 
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True)

    
    best_accuracy = 0.0
    best_model_path = f"model/{model_name}/best_model"
    best_valid_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 5


    if local_rank == 0:
        print(f"Total Epoch: {args.epochs}")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            # print(f"---------------Epoch {epoch+1}---------------")
            logging.info(f"---------------Epoch {epoch+1}---------------")

        train_loss, accuracy = train(model, train_loader, optimizer, device, scaler, local_rank)
        if local_rank == 0:
            # print(f"Train set: Train Loss:{train_loss}, Accuracy: {accuracy}")
            logging.info(f"Train set: Train Loss:{train_loss}, Accuracy: {accuracy}")
        
        valid_loss, acc = evaluate(model, val_loader, device, local_rank)
        scheduler.step(valid_loss)

        if local_rank == 0:
            # print(f"Valid set: Valid Loss: {valid_loss}, Accuracy: {acc}")
            logging.info(f"Valid set: Valid Loss: {valid_loss}, Accuracy: {acc}")

            # if acc > best_accuracy:
            #     best_accuracy = acc
            #     model_to_save = model.module if hasattr(model, 'module') else model  # 处理DDP封装
            #     model_to_save.save_pretrained(best_model_path)
            #     # print(f"New best model with accuracy {best_accuracy} saved to {best_model_path}")
            #     logging.info(f"New best model with accuracy {best_accuracy} saved to {best_model_path}")
            
            #early stop strategy
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                # print(f"New best model with loss {best_valid_loss} saved to {best_model_path}")
                logging.info(f"New best model with loss {best_valid_loss} saved to {best_model_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                # if early_stopping_counter >= early_stopping_patience:
                #     # print(f"Early stopping triggered after {epoch + 1} epochs.")
                #     logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                #     break
            
        # 使用torch.distributed.broadcast来同步早停决定
        early_stopping_decision = torch.tensor(early_stopping_counter > early_stopping_patience, dtype=torch.bool).to(device)
        torch.distributed.broadcast(early_stopping_decision, src=0)  # src=0 表示主进程

        if early_stopping_decision.item():
            if local_rank == 0:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break


    # 测试集评估
    test_loss, acc = evaluate(model, test_loader, device, local_rank)
    if local_rank == 0:
        # print(f"Test set: Test Loss: {test_loss}, Accuracy: {acc}")
        logging.info(f"Test set: Test Loss: {test_loss}, Accuracy: {acc}")


if __name__ == "__main__":
    main()
