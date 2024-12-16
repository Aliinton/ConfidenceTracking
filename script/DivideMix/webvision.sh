python train_dividemix.py --data_dir 'path_to_data' --dataset webvision --experiment_name webvision \
--arc InceptionResNetV2 --n_epoch 100 --batch_size 32 --lr 0.01 --weight_decay 0.0005 --milestones 50 \
--warmup_epoch 1 --lambda_u 0 --mixup_alpha 0.5 --seed 123 --save_dir ./save/DivideMix

