#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/CE/cifar-10.sh symmetric 0.2 200 $DATA_DIR
bash ./script/CE/cifar-10.sh symmetric 0.5 200 $DATA_DIR
bash ./script/CE/cifar-10.sh asymmetric 0.4 200 $DATA_DIR

bash ./script/CE/cifar-100.sh symmetric 0.2 200 $DATA_DIR
bash ./script/CE/cifar-100.sh symmetric 0.5 200 $DATA_DIR
bash ./script/CE/cifar-100.sh asymmetric 0.4 200 $DATA_DIR

