#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/AUM2_CT/cifar-10.sh clean 0.00 30 $DATA_DIR 0.05 0.01
bash ./script/AUM2_CT/cifar-10.sh aggre 0.09 30 $DATA_DIR 0.05 0.01
bash ./script/AUM2_CT/cifar-10.sh rand1 0.18 30 $DATA_DIR 0.05 0.01
bash ./script/AUM2_CT/cifar-10.sh rand2 0.18 30 $DATA_DIR 0.05 0.01
bash ./script/AUM2_CT/cifar-10.sh rand3 0.18 30 $DATA_DIR 0.05 0.01
bash ./script/AUM2_CT/cifar-10.sh worst 0.4 30 $DATA_DIR 0.05 0.01

bash ./script/AUM2_CT/cifar-100.sh clean 0.00 30 $DATA_DIR 0.05 0.01
bash ./script/AUM2_CT/cifar-100.sh noisy 0.4 30 $DATA_DIR 0.00 0.01
