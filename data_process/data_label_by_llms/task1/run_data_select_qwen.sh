export DASHSCOPE_API_KEY=sk-ee18c7ad32054d0884066409d42a2e83


DATA_FILE="processed_data/classification_data/soc_texts.txt"
TARGET_COUNT=2000
ALL_RESULTS_PATH="processed_data/classification_data/hq_data_qwen/soc_texts_all_results.jsonl"
HIGH_QUALITY="processed_data/classification_data/hq_data_qwen/soc_texts_high_quality_data.jsonl"


python data_select_useLLM.py \
    --data_file $DATA_FILE\
    --target_count $TARGET_COUNT\
    --all_results_path $ALL_RESULTS_PATH\
    --high_quality_results_path $HIGH_QUALITY