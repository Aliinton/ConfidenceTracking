python train_dividemix.py --data_dir 'path_to_data' --dataset food101n --experiment_name food101n \
--arc resnet50 --n_epoch 30 --batch_size 32 --lr 0.002 --weight_decay 0.0005 --milestones 10 20 \
--warmup_epoch 1 --lambda_u 0 --mixup_alpha 0.5 --save_dir ./save/DivideMix+CT --ct_refine --alpha 0.10