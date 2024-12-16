#!/bin/bash
NOISE_TYPE=$1
NOISE_RATE=$2
WARMUP=$3
DATA_DIR=$4
P_THRESHOLD=$5
ALPHA=$6
EXPERIMENT_NAME="CIFAR10_${NOISE_TYPE}_${NOISE_RATE}_report"
SAVE_DIR="save/GMM+CT/threshold_${P_THRESHOLD}/alpha_${ALPHA}"
for SEED in 1 2 3 4 5
do
  python train.py --experiment_name $EXPERIMENT_NAME --dataset cifar-10 --seed $SEED \
  --noise_type $NOISE_TYPE --noise_rate $NOISE_RATE --n_epoch 150 --data_dir $DATA_DIR \
  --batch_size 128 --lr 0.02 --lr_scheduler MultiStepLR --milestones 80 --p_threshold $P_THRESHOLD \
  --gamma 0.1 --momentum 0.9 --weight_decay 0.001 --early_stopping_patient 2000 \
  --warmup_epoch $WARMUP --save_dir $SAVE_DIR --selector 'GMM+CT' --alg dynamic_sample_selection --alpha $ALPHA
done
