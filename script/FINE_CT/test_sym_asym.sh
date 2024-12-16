#!/bin/bash
DATA_DIR='path_to_data'
bash ./script/FINE_CT/cifar-10.sh symmetric 0.2 30 $DATA_DIR 0.01
bash ./script/FINE_CT/cifar-10.sh symmetric 0.5 30 $DATA_DIR 0.01
bash ./script/FINE_CT/cifar-10.sh asymmetric 0.4 30 $DATA_DIR 0.01

bash ./script/FINE_CT/cifar-100.sh symmetric 0.2 30 $DATA_DIR 0.01
bash ./script/FINE_CT/cifar-100.sh symmetric 0.5 30 $DATA_DIR 0.01
bash ./script/FINE_CT/cifar-100.sh asymmetric 0.4 30 $DATA_DIR 0.01

