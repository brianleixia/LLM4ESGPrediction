#!/bin/bash

# 定义顶层目录名称，可以是相对路径也可以是绝对路径
# top_dir="four_class"
top_dir="nine_class"

# 定义子目录列表
sub_dirs=(
    # "llama2"
    # "llama2-classificaion-freeze"
    # "llama2-esg-freeze"
    # "llama2-fin-esg-freeze"
    "llama2-classificaion-lora"
)

# 定义需要遍历的内部文件夹名称
inner_dirs=(
    "zero-shot"
    "zero-shot-cot"
    "one-shot"
    "one-shot-cot"
    "ICL"
    "ICL-cot"
)

CSV_FILE="nine_class_eval.csv"
# 遍历子目录
for sub_dir in "${sub_dirs[@]}"; do
    # 遍历内部文件夹
    for inner_dir in "${inner_dirs[@]}"; do
        # 构建完整的目录路径
        full_dir="${top_dir}/${sub_dir}/${inner_dir}"
        # 检查目录是否存在
        if [ -d "$full_dir" ]; then
            # 检查predicted_results.jsonl文件是否存在
            predicted_results_file="${full_dir}/predicted_results.jsonl"
            # predicted_results_file="${full_dir}/predictions.jsonl"
            if [ -f "$predicted_results_file" ]; then
                # 调用Python评估脚本
                python evaluate_llm.py "$full_dir" --csv $CSV_FILE
                echo "评估完成: ${predicted_results_file}"
            else
                echo "文件不存在，跳过评估: ${predicted_results_file}"
            fi
        else
            echo "目录不存在，跳过: ${full_dir}"
        fi
    done
done
