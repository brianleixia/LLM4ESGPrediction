export DASHSCOPE_API_KEY=sk-ee18c7ad32054d0884066409d42a2e83


DATA_FILE="env.txt"
TARGET_COUNT=5000

ALL_RESULTS_PATH="processed_data/high_quality_data/glm/env_all_results.jsonl"
HIGH_QUALITY="processed_data/high_quality_data/glm/env_high_quality_data.jsonl"

# DATA_FILE="gov.txt"
# ALL_RESULTS_PATH="processed_data/high_quality_data/glm/gov_all_results.jsonl"
# HIGH_QUALITY="processed_data/high_quality_data/glm/gov_high_quality_data.jsonl"

# DATA_FILE="soc.txt"
# ALL_RESULTS_PATH="processed_data/high_quality_data/glm/soc_all_results.jsonl"
# HIGH_QUALITY="processed_data/high_quality_data/glm/soc_high_quality_data.jsonl"


python data_select_use_glm.py \
    --data_file $DATA_FILE\
    --target_count $TARGET_COUNT\
    --all_results_path $ALL_RESULTS_PATH\
    --high_quality_results_path $HIGH_QUALITY