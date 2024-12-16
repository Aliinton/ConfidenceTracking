python train_dividemix.py --data_dir 'path_to_data' --dataset clothing1m --experiment_name clothing1m \
--arc resnet50 --n_epoch 80 --batch_size 32 --lr 0.002 --weight_decay 0.001 --milestones 40 --num_batches 1000 \
--warmup_epoch 1 --lambda_u 0 --mixup_alpha 0.5 --save_dir ./save/DivideMix+CT --seed 123 --ct_refine --alpha 0.01
