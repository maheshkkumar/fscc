#!/usr/bin/env bash

DATASET=""   # name of the dataset
DATA_PATH="" # path of the dataset
LOG_DIR=""   # path of the log directory

NUM_TASKS=8
NUM_INSTANCES=5
META_BATCHSIZE=5
BASE_BATCHSIZE=1
META_LR=0.001
BASE_LR=0.001
EPOCHS=1500
BASE_UPDATES=10
EXPERIMENT=1
TRAIN_PREFIX="python train.py --dataset $DATASET"

CMD="${TRAIN_PREFIX} --data_path ${DATA_PATH} --num_tasks ${NUM_TASKS} --num_instances ${NUM_INSTANCES} --meta_batch ${META_BATCHSIZE} --base_batch ${BASE_BATCHSIZE} --meta_lr ${META_LR} --base_lr ${BASE_LR} --epochs ${EPOCHS} --base_updates ${BASE_UPDATES} --experiment ${EXPERIMENT} --log ${LOG_DIR}"

echo $CMD
eval $CMD
