# MODEL_PATH="../open-source-llms/Llama-2-7b-chat-hf"
# MODEL_PATH="../LLaMA-Factory/llama2_7b_esg"
# CHAT_TEMP_PATH="vllm/examples/template_chatml.jinja"

# MODEL_PATH="../LLaMA-Factory/llama2_7b_classification_freeze_merge"
# MODEL_PATH="../LLaMA-Factory/llama2_7b_esg_lora_merge_classification_freeze_merge"
MODEL_PATH="../LLaMA-Factory/llama2_7b_fin_pt_FinEsgSFT_classification_freeze_merge"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 8

# --chat-template $CHAT_TEMP_PATH