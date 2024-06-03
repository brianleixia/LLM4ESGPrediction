MODEL_NAME='output/bert/bert-epoch25'
MODEL_TYPE='bert'
BATCH_SIZE=32
EPOCHS=50
LR=3e-6


python3 -m torch.distributed.launch --nproc_per_node=8 finetune_ddp.py \
    --model_name $MODEL_NAME\
    --model_type $MODEL_TYPE\
    --batch_size $BATCH_SIZE\
    --epochs $EPOCHS\
    --learning_rate $LR\