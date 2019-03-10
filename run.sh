#!/bin/bash
set -x

SCHEDULE=schedule.json
TRY_NEW=True
USE_TENSORBOARD=True
USE_GPU=True
NUM_TRAINING_SET=50000
BATCH_SIZE=128
PRINT_EVERY=100

python main.py \
    --mode='train' \
    --schedule=$SCHEDULE \
    --use-tb=$USE_TENSORBOARD \
    --use-gpu=$USE_GPU \
    --try-new=$TRY_NEW \
    --num-train=$NUM_TRAINING_SET \
    --batch-size=$BATCH_SIZE \
    --print-every=$PRINT_EVERY
