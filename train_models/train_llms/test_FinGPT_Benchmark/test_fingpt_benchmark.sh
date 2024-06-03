BASE_MODEL="/apdcephfs/share_916081/shared_info/brianleixia/project/open-source-llms/Llama-2-7b-chat-hf"

# BASE_MODEL="/apdcephfs/share_916081/shared_info/brianleixia/project/LLaMA-Factory/llama2_7b_fin_pt_lora_merge"
# PEFT_MODEL="/apdcephfs/share_916081/shared_info/brianleixia/project/LLaMA-Factory/llama2_7b_fin_pt_lora_merge_FinEsgSFT_lora"

# fpb,fiqa,tfns,headline,ner,re
python benchmarks.py \
--dataset headline \
--base_model $BASE_MODEL \
--batch_size 64 \
--max_length 512


BASE_MODEL="/apdcephfs/share_916081/shared_info/brianleixia/project/LLaMA-Factory/llama2_7b_fin_pt_FinEsgSFT_classification_freeze_merge"
python benchmarks.py \
--dataset headline \
--base_model $BASE_MODEL \
--batch_size 64 \
--max_length 512


BASE_MODEL="/apdcephfs/share_916081/shared_info/brianleixia/project/LLaMA-Factory/llama2_7b_fin_pt_lora_merge"
PEFT_MODEL="/apdcephfs/share_916081/shared_info/brianleixia/project/LLaMA-Factory/llama2_7b_fin_pt_lora_merge_FinEsgSFT_lora"
python benchmarks.py \
--dataset headline \
--base_model $BASE_MODEL \
--peft_model $PEFT_MODEL \
--batch_size 64 \
--max_length 512