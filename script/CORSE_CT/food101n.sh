python train_CORSE.py --data_dir 'path_to_data' --dataset food101n --experiment_name food101n \
--arc resnet50 --n_epoch 30 --batch_size 64 --lr 0.002 --weight_decay 0.0005 --milestones 10 20 \
--warmup_epoch 1 --beta 2.0 --ct_refine --alpha 0.1 --save_dir ./save/CORSE+CT/
