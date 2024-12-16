python train_CORSE.py --data_dir 'path_to_data' --dataset webvision --experiment_name webvision \
--arc InceptionResNetV2 --n_epoch 100 --batch_size 64 --lr 0.01 --weight_decay 0.0005 --milestones 50 \
--warmup_epoch 1 --seed 123 --beta 2.0 --ct_refine --alpha 0.1 --save_dir ./save/CORSE+CT/