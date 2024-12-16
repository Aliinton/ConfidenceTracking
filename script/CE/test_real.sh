#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/CE/cifar-10.sh clean 0.0 200 $DATA_DIR
bash ./script/CE/cifar-10.sh aggre 0.09 200 $DATA_DIR
bash ./script/CE/cifar-10.sh rand1 0.18 200 $DATA_DIR
bash ./script/CE/cifar-10.sh rand2 0.18 200 $DATA_DIR
bash ./script/CE/cifar-10.sh rand3 0.18 200 $DATA_DIR
bash ./script/CE/cifar-10.sh worst 0.4 200 $DATA_DIR

bash ./script/CE/cifar-100.sh clean 0.0 200 $DATA_DIR
bash ./script/CE/cifar-100.sh noisy 0.4 200 $DATA_DIR

