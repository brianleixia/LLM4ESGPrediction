# MODEL_PATH="../open-source-llms/Llama-2-7b-chat-hf"

# MODEL_PATH="llama2_7b_fin_pt_FinEsgSFT"
# OUTPUT_FOLDER="llama2_7b_fin_pt_FinEsgSFT_classification_freeze"

# MODEL_PATH="llama2_7b_esg_lora_merge"
# OUTPUT_FOLDER="llama2_7b_esg_lora_merge_classification_freeze"

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
#     --dataset esg_classification \
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


# MODEL_PATH="../open-source-llms/Llama-2-7b-chat-hf"
# OUTPUT_FOLDER="llama2_7b_classification_freeze"
# accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path $MODEL_PATH \
#     --dataset esg_classification \
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


# OUTPUT_FOLDER="llama2_7b_esg_lora_merge_classification_freeze_kagpu"
# accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path $MODEL_PATH \
#     --dataset esg_classification \
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
#     --num_train_epochs 500.0 \
#     --plot_loss \
#     --overwrite_output_dir \
#     --cutoff_len 2048 \
#     --fp16

# accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path $MODEL_PATH \
#     --dataset fin_esg_sft \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
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


MODEL_PATH="../open-source-llms/phi-1_5"
OUTPUT_FOLDER="phi-1_5_classification"
accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset esg_classification \
    --template default \
    --finetuning_type full \
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
    --cutoff_len 2048 \
    --fp16


# OUTPUT_FOLDER="llama2_7b_classification_freeze_kagpu2"
# accelerate launch --config_file $ACCELERATE_CONFIG src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path $MODEL_PATH \
#     --dataset esg_classification \
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
#     --num_train_epochs 500.0 \
#     --plot_loss \
#     --overwrite_output_dir \
#     --cutoff_len 2048 \
#     --fp16