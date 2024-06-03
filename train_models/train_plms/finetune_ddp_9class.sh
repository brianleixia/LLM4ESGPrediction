MODEL_NAME='output/roberta/roberta-epoch25'
MODEL_TYPE='roberta'
BATCH_SIZE=32
EPOCHS=50
LR=1.15e-6


python3 -m torch.distributed.launch --nproc_per_node=8 finetune_ddp_9class.py \
    --model_name $MODEL_NAME\
    --model_type $MODEL_TYPE\
    --batch_size $BATCH_SIZE\
    --epochs $EPOCHS\
    --learning_rate $LR\