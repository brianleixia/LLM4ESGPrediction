MODEL_PATH="../open-source-llms/Llama-2-7b-chat-hf"
# ADAPTER_FOLDER="llama2_7b_esg_lora"
# EXPORT_DIR="llama2_7b_esg_lora_merge"

# ADAPTER_FOLDER="llama2_7b_fin_pt_lora"
# EXPORT_DIR="llama2_7b_fin_pt_lora_merge"

# ADAPTER_FOLDER="llama2_7b_fin_pt_FinEsgSFT_classification_lora"
# EXPORT_DIR="llama2_7b_fin_pt_FinEsgSFT_classification_lora_merge"

ADAPTER_FOLDER="llama2_7b_fin_pt_FinEsgSFT_classification_freeze"
EXPORT_DIR="llama2_7b_fin_pt_FinEsgSFT_classification_freeze_merge"

python src/export_model.py \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $ADAPTER_FOLDER \
    --template default \
    --finetuning_type freeze \
    --export_dir $EXPORT_DIR \
    --export_size 2 \
    --export_legacy_format False

# python src/export_model.py \
#     --model_name_or_path $MODEL_PATH \
#     --adapter_name_or_path $ADAPTER_FOLDER \
#     --template default \
#     --finetuning_type lora \
#     --export_dir $EXPORT_DIR \
#     --export_size 2 \
#     --export_legacy_format False