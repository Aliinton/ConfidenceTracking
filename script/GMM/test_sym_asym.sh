#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/GMM/cifar-10.sh symmetric 0.2 30 $DATA_DIR 0.5
bash ./script/GMM/cifar-10.sh symmetric 0.5 30 $DATA_DIR 0.5
bash ./script/GMM/cifar-10.sh asymmetric 0.4 30 $DATA_DIR 0.5

bash ./script/GMM/cifar-100.sh symmetric 0.2 30 $DATA_DIR 0.5
bash ./script/GMM/cifar-100.sh symmetric 0.5 30 $DATA_DIR 0.5
bash ./script/GMM/cifar-100.sh asymmetric 0.4 30 $DATA_DIR 0.9

