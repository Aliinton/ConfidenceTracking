#!/bin/bash
NOISE_TYPE=$1
NOISE_RATE=$2
WARMUP=$3
DATA_DIR=$4
DELTA=$5
EXPERIMENT_NAME="CIFAR10_${NOISE_TYPE}_${NOISE_RATE}_report"
SAVE_DIR="save/report/L2D/delta${DELTA}"
for SEED in 1 2 3 4 5
do
  python ./L2D/train.py --experiment_name $EXPERIMENT_NAME --dataset cifar-10 --seed $SEED \
  --noise_type $NOISE_TYPE --noise_rate $NOISE_RATE --n_epoch 150 \
  --batch_size 128 --lr 0.02 --lr_scheduler MultiStepLR --milestones 80 \
  --gamma 0.1 --momentum 0.9 --weight_decay 0.001 \
  --delta $DELTA --data_dir $DATA_DIR --save_dir $SAVE_DIR
done
