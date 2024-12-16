#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/GMM/cifar-10.sh instance 0.1 30 $DATA_DIR 0.5
bash ./script/GMM/cifar-10.sh instance 0.3 30 $DATA_DIR 0.7
bash ./script/GMM/cifar-10.sh instance 0.5 30 $DATA_DIR 0.9

bash ./script/GMM/cifar-100.sh instance 0.1 30 $DATA_DIR 0.5
bash ./script/GMM/cifar-100.sh instance 0.3 30 $DATA_DIR 0.5
bash ./script/GMM/cifar-100.sh instance 0.5 30 $DATA_DIR 0.9
