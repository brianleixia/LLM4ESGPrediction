#!/bin/bash

# 定义一个函数来处理中断信号
handle_interrupt() {
    echo "Interrupted, stopping all subprocesses..."
    kill 0  # 杀死当前脚本的所有子进程
    exit 1
}

# 捕获中断信号（Ctrl+C）
trap 'handle_interrupt' INT

# 分类变量，可以设置为 gov, env 或 soc
CATEGORY="soc"

DATA_FOLDER="split_data/${CATEGORY}"
OUTPUT_FOLDER="output/${CATEGORY}"
USAGE_FOLDER="usage/${CATEGORY}"

# 创建输出目录，如果不存在的话
mkdir -p $OUTPUT_FOLDER
mkdir -p $USAGE_FOLDER


echo "Start process..."
# 遍历目录中的所有分割文件
for DATA_FILE in ${DATA_FOLDER}/*.jsonl; do
    # 获取分割文件的基本名（不带路径）
    BASE_NAME=$(basename -- "$DATA_FILE")

    # 创建一个唯一的结果文件路径
    ALL_RESULTS_PATH="${OUTPUT_FOLDER}/${BASE_NAME}_results.jsonl"
    USAGE_FILE="${USAGE_FOLDER}/${BASE_NAME}_usage.jsonl"

    echo "Processing file: $DATA_FILE"
    echo "Results will be saved to: $ALL_RESULTS_PATH"

    python socdata_classify_use_gpt3.py \
        --data_file $DATA_FILE \
        --all_results_path $ALL_RESULTS_PATH \
        --usage_file $USAGE_FILE &
done

wait

echo "All processes have completed."
