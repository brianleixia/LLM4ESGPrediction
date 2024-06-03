#!/bin/bash

# 定义训练参数
MODEL_TYPE="roberta"  # DistilRoBERTa 使用的是 RoBERTa 架构
MODEL_NAME="open-source-models/distilroberta-base"  # 指向本地模型目录
TRAIN_FILE="data/train.txt"
VALID_FILE="data/valid.txt"
OUTPUT_DIR="output/distilroberta/distilroberta-epoch25"

# 训练参数
BATCH_SIZE=16
EVAL_BATCH_SIZE=16
# NUM_EPOCHS=5
NUM_EPOCHS=25
SAVE_STEPS=10000
SEED=42

# 把根据steps保存的逻辑换成了epoch
# --save_steps $SAVE_STEPS \


# 运行MLM训练脚本
python3 -m torch.distributed.launch --nproc_per_node=8 run_mlm.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --output_dir $OUTPUT_DIR \
    --line_by_line \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --num_train_epochs $NUM_EPOCHS \
    --seed $SEED \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --weight_decay 0.01 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --fp16 

