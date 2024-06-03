DATA_FILE="gov.txt"
TARGET_COUNT=5000
ALL_RESULTS_PATH="processed_data/high_quality_data/chatgpt3/gov_all_results.jsonl"
HIGH_QUALITY="processed_data/high_quality_data/chatgpt3/gov_high_quality_data.jsonl"
USAGE_FILE="usage/gov.jsonl"

# DATA_FILE="env.txt"
# TARGET_COUNT=5000
# ALL_RESULTS_PATH="processed_data/high_quality_data/chatgpt3/env_all_results.jsonl"
# HIGH_QUALITY="processed_data/high_quality_data/chatgpt3/env_high_quality_data.jsonl"
# USAGE_FILE="usage/gov.jsonl"


python data_select_use_gpt.py \
    --data_file $DATA_FILE \
    --target_count $TARGET_COUNT \
    --all_results_path $ALL_RESULTS_PATH \
    --high_quality_results_path $HIGH_QUALITY \
    --usage_file $USAGE_FILE