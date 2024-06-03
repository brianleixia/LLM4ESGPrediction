#!/bin/bash

# Function to run predictions
run_predictions() {
    local model_base_path=$1
    local data_type=$2
    local model_subdir=$3

    local model_path="${model_base_path}/${model_subdir}/${data_type}/best_model"
    local output_dir="predictions/${model_subdir}/${data_type}"

    # Run the Python script with the specified parameters
    echo "Running predictions for model in ${model_path} on ${data_type} data..."
    python eval_auto.py --model_path "$model_path" --output_dir "$output_dir" --data_type "$data_type" 
}

# Fine-tuned Models
model_base_path='model'
run_predictions "$model_base_path" "env" "roberta/roberta-epoch25"
run_predictions "$model_base_path" "gov" "roberta/roberta-epoch25"
run_predictions "$model_base_path" "soc" "roberta/roberta-epoch25"

run_predictions "$model_base_path" "env" "distilroberta/distilroberta-epoch25"
run_predictions "$model_base_path" "gov" "distilroberta/distilroberta-epoch25"
run_predictions "$model_base_path" "soc" "distilroberta/distilroberta-epoch25"

run_predictions "$model_base_path" "env" "bert/bert-epoch25"
run_predictions "$model_base_path" "gov" "bert/bert-epoch25"
run_predictions "$model_base_path" "soc" "bert/bert-epoch25"

run_predictions "$model_base_path" "env" "roberta-base"
run_predictions "$model_base_path" "gov" "roberta-base"
run_predictions "$model_base_path" "soc" "roberta-base"

run_predictions "$model_base_path" "env" "distilroberta-base"
run_predictions "$model_base_path" "gov" "distilroberta-base"
run_predictions "$model_base_path" "soc" "distilroberta-base"

run_predictions "$model_base_path" "env" "bert-base-uncased"
run_predictions "$model_base_path" "gov" "bert-base-uncased"
run_predictions "$model_base_path" "soc" "bert-base-uncased"

# Wait for all background jobs to finish
wait

echo "All predictions completed."
