#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/DIST/cifar-10.sh symmetric 0.2 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-10.sh symmetric 0.5 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-10.sh asymmetric 0.4 30 $DATA_DIR 0.95

bash ./script/DIST/cifar-100.sh symmetric 0.2 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-100.sh symmetric 0.5 30 $DATA_DIR 0.99
bash ./script/DIST/cifar-100.sh asymmetric 0.4 30 $DATA_DIR 0.95
