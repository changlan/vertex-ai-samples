#!/bin/sh

# Change the following variables as needed:
NUM_GPUS_PER_WORKER=${NUM_GPUS_PER_WORKER:-$(nvidia-smi -L | wc -l)}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-2222}

echo "Number of workers: $WORLD_SIZE"
echo "Worker rank: $RANK"
echo "GPUs per worker: $NUM_GPUS_PER_WORKER"

TASK_NAME=mnli

python -m torch.distributed.launch \
  --nproc_per_node $NUM_GPUS_PER_WORKER --nnodes $WORLD_SIZE --node_rank $RANK \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  run_glue.py \
  --model_name_or_path bert-large-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  $@