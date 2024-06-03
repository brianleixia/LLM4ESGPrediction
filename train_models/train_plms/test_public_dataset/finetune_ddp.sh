# MODEL_PATH='../output/roberta/roberta-epoch25'
# MODEL_NAME='roberta/roberta-epoch25/env'
# MODEL_PATH='../output/roberta/roberta-epoch25'
# MODEL_NAME='roberta/roberta-epoch25/gov'
# MODEL_PATH='../output/roberta/roberta-epoch25'
# MODEL_NAME='roberta/roberta-epoch25/soc'

# MODEL_PATH='../output/distilroberta/distilroberta-epoch25'
# MODEL_NAME='distilroberta/distilroberta-epoch25/env'
# MODEL_PATH='../output/distilroberta/distilroberta-epoch25'
# MODEL_NAME='distilroberta/distilroberta-epoch25/gov'
# MODEL_PATH='../output/distilroberta/distilroberta-epoch25'
# MODEL_NAME='distilroberta/distilroberta-epoch25/soc'

# MODEL_PATH='../output/bert/bert-epoch25'
# MODEL_NAME='bert/bert-epoch25/env'
# MODEL_PATH='../output/bert/bert-epoch25'
# MODEL_NAME='bert/bert-epoch25/gov'
# MODEL_PATH='../output/bert/bert-epoch25'
# MODEL_NAME='bert/bert-epoch25/soc'


# Base Models:
# MODEL_PATH='../open-source-models/roberta-base'
# MODEL_NAME='roberta-base/env'
# MODEL_PATH='../open-source-models/roberta-base'
# MODEL_NAME='roberta-base/gov'
# MODEL_PATH='../open-source-models/roberta-base'
# MODEL_NAME='roberta-base/soc'

# MODEL_PATH='../open-source-models/distilroberta-base'
# MODEL_NAME='distilroberta-base/env'
# MODEL_PATH='../open-source-models/distilroberta-base'
# MODEL_NAME='distilroberta-base/gov'
MODEL_PATH='../open-source-models/distilroberta-base'
MODEL_NAME='distilroberta-base/soc'

# MODEL_PATH='../open-source-models/bert-base-uncased'
# MODEL_NAME='bert-base-uncased/env'
# MODEL_PATH='../open-source-models/bert-base-uncased'
# MODEL_NAME='bert-base-uncased/gov'
# MODEL_PATH='../open-source-models/bert-base-uncased'
# MODEL_NAME='bert-base-uncased/soc'


MODEL_TYPE='roberta'
# MODEL_TYPE='bert'
BATCH_SIZE=32
EPOCHS=50
LR=3e-6


python3 -m torch.distributed.launch --nproc_per_node=4 finetune_ddp.py \
    --model_path $MODEL_PATH\
    --model_name $MODEL_NAME\
    --model_type $MODEL_TYPE\
    --batch_size $BATCH_SIZE\
    --epochs $EPOCHS\
    --learning_rate $LR\