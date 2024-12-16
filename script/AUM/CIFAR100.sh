#!/bin/bash
NOISE_TYPE=$1
NOISE_RATE=$2
DATA_DIR=$3
EXPERIMENT_NAME="CIFAR100_${NOISE_TYPE}_${NOISE_RATE}"
for SEED in 1 2 3 4 5
do
  python train.py --experiment_name $EXPERIMENT_NAME --dataset cifar-100 --seed $SEED --save_dir save/AUM \
  --noise_type $NOISE_TYPE --noise_rate $NOISE_RATE --n_epoch 150 \
  --batch_size 128 --lr 0.02 --lr_scheduler MultiStepLR --milestones 80 \
  --gamma 0.1 --momentum 0.9 --weight_decay 0.001 --alg AUM --data_dir $DATA_DIR
done
