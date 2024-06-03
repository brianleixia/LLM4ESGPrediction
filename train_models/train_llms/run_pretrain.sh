MODEL_PATH="../open-source-llms/Llama-2-7b-chat-hf"
# OUTPUT_FOLDER="llama2_7b_test_alpaca_gpt4_en2"
OUTPUT_FOLDER="llama2_7b_fin_pt_lora"
OUTPUT_FOLDER2="llama2_7b_fin_pt_lora_kagpu"
ACCELERATE_CONFIG="accelerate_config.yaml"

# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path $MODEL_PATH \
#     --dataset alpaca_gpt4_en \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir $OUTPUT_FOLDER \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --plot_loss \
#     --overwrite_output_dir \
#     --fp16

	
# accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path $MODEL_PATH \
#     --dataset esg_class_sft \
#     --template default \
#     --finetuning_type freeze \
#     --output_dir $OUTPUT_FOLDER \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 100 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --plot_loss \
#     --overwrite_output_dir \
#     --cutoff_len 2048 \
#     --fp16


accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
    --stage pt \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset financial_text_small \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $OUTPUT_FOLDER \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --fp16


# accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
#     --stage pt \
#     --do_train \
#     --model_name_or_path $MODEL_PATH \
#     --dataset financial_text \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir $OUTPUT_FOLDER2 \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 100 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 150.0 \
#     --plot_loss \
#     --overwrite_output_dir \
#     --fp16
