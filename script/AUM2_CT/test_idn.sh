#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/AUM2_CT/cifar-10.sh instance 0.1 30 $DATA_DIR 0.10 0.50
bash ./script/AUM2_CT/cifar-10.sh instance 0.3 30 $DATA_DIR 0.10 0.50
bash ./script/AUM2_CT/cifar-10.sh instance 0.5 30 $DATA_DIR 0.10 0.50

bash ./script/AUM2_CT/cifar-100.sh instance 0.1 30 $DATA_DIR 0.10 0.50
bash ./script/AUM2_CT/cifar-100.sh instance 0.3 30 $DATA_DIR 0.10 0.50
bash ./script/AUM2_CT/cifar-100.sh instance 0.5 30 $DATA_DIR 0.10 0.10
