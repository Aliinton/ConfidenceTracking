#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/L2D/cifar-10.sh symmetric 0.2 30 $DATA_DIR 0.00
bash ./script/L2D/cifar-10.sh symmetric 0.5 30 $DATA_DIR 0.05
bash ./script/L2D/cifar-10.sh asymmetric 0.4 30 $DATA_DIR 0.00
bash ./script/L2D/cifar-10.sh worst 0.4 30 $DATA_DIR 0.10

bash ./script/L2D/cifar-100.sh symmetric 0.2 30 $DATA_DIR 0.05
bash ./script/L2D/cifar-100.sh symmetric 0.5 30 $DATA_DIR 0.10
bash ./script/L2D/cifar-100.sh asymmetric 0.4 30 $DATA_DIR 0.05
bash ./script/L2D/cifar-100.sh noisy 0.4 30 $DATA_DIR 0.00
