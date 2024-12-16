#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/AUM/CIFAR10.sh symmetric 0.2 $DATA_DIR
bash ./script/AUM/CIFAR10.sh symmetric 0.5 $DATA_DIR
bash ./script/AUM/CIFAR10.sh asymmetric 0.4 $DATA_DIR
bash ./script/AUM/CIFAR10.sh worst 0.0 $DATA_DIR

bash ./script/AUM/CIFAR100.sh symmetric 0.2 $DATA_DIR
bash ./script/AUM/CIFAR100.sh symmetric 0.5 $DATA_DIR
bash ./script/AUM/CIFAR100.sh asymmetric 0.4 $DATA_DIR
bash ./script/AUM/CIFAR100.sh noisy 0.0 $DATA_DIR