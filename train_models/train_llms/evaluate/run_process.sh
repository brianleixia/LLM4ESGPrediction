#!/bin/bash

# 定义顶层目录
# top_dir="four_class"
top_dir="nine_class"

# 定义子目录列表
sub_dirs=(
    "llama2"
    "llama2-classificaion-freeze"
    # "llama2-classificaion-lora"
    "llama2-esg-freeze"
    "llama2-fin-esg-freeze"
)

# 定义模式目录列表
pattern_dirs=(
    "zero-shot"
    "zero-shot-cot"
    "one-shot"
    "one-shot-cot"
    "ICL"
    "ICL-cot"
)

# 遍历子目录
for sub_dir in "${sub_dirs[@]}"; do
    # 遍历模式目录
    for pattern_dir in "${pattern_dirs[@]}"; do
        # 构建完整的目录路径
        parent_folder="${top_dir}/${sub_dir}/${pattern_dir}"

        # 指定输入文件、匹配输出文件和未匹配输出文件的路径
        input_file="${parent_folder}/output/predictions.jsonl"
        matched_output_file="${parent_folder}/matched_output_file.jsonl"
        unmatched_output_file="${parent_folder}/unmatched_output_file.jsonl"

        # 确保输出目录存在
        mkdir -p "${parent_folder}/output"

        # 检查输入文件是否存在
        if [ -f "$input_file" ]; then
            # 调用Python脚本处理每个文件夹中的predictions.jsonl文件
            python process_response.py "$input_file" "$matched_output_file" "$unmatched_output_file"
            echo "处理完成: ${input_file}"
        else
            echo "文件不存在，跳过: ${input_file}"
        fi
    done
done
