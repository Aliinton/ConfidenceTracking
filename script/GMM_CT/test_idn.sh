#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/GMM_CT/cifar-10.sh instance 0.1 30 $DATA_DIR 0.95 0.50
bash ./script/GMM_CT/cifar-10.sh instance 0.3 30 $DATA_DIR 0.95 0.50
bash ./script/GMM_CT/cifar-10.sh instance 0.5 30 $DATA_DIR 0.99 0.01

bash ./script/GMM_CT/cifar-100.sh instance 0.1 30 $DATA_DIR 0.99 0.50
bash ./script/GMM_CT/cifar-100.sh instance 0.3 30 $DATA_DIR 0.70 0.10
bash ./script/GMM_CT/cifar-100.sh instance 0.5 30 $DATA_DIR 0.95 0.01
