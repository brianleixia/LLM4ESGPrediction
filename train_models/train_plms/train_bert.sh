#!/bin/bash

# 定义训练参数
MODEL_TYPE="bert"
MODEL_NAME="open-source-models/bert-base-uncased"  # 指向本地模型目录
TRAIN_FILE="corpus.txt"
OUTPUT_DIR="output/bert"

# 训练参数
BATCH_SIZE=16
NUM_EPOCHS=3
SAVE_STEPS=10000
SEED=42

# 运行MLM训练脚本
python3 run_mlm.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --line_by_line \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --max_seq_length 512 \
    --overwrite_output_dir
