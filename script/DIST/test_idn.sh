#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/DIST/cifar-10.sh instance 0.1 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-10.sh instance 0.3 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-10.sh instance 0.5 30 $DATA_DIR 0.95

bash ./script/DIST/cifar-100.sh instance 0.1 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-100.sh instance 0.3 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-100.sh instance 0.5 30 $DATA_DIR 0.95
