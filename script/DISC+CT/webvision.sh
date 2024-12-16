python train_disc.py --alg DISC --data_dir 'path_to_data' --experiment_name webvision --dataset webvision \
--arc InceptionResNetV2 --n_epoch 100 --batch_size 32 --lr 0.2 --weight_decay 0.0005 --milestones 50 80 \
--warmup_epoch 15 --decay 0.99 --sigma 0.3 --seed 1 --save_dir save/DISC --ct_refine --alpha 0.10
