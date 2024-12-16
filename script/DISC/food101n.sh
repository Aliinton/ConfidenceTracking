python train_disc.py --alg DISC --data_dir 'path_to_data' --experiment_name food101n --dataset food101n \
--arc resnet50 --n_epoch 30 --batch_size 32 --lr 0.01 --weight_decay 0.0005 --milestones 10 20 \
--warmup_epoch 5 --decay 0.99 --sigma 0.3 --seed 1 --save_dir save/DISC
