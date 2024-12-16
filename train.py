import os
import json
import random
import argparse
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloader.dataloaders import *
from algorithm import AUM, CoTeaching, DSS, CNLCU, FINE, HMW


NUM_CLASSES = {'cifar-10': 10, 'cifar-100': 100, 'cifar-10n': 10, 'cifar-100n': 100, 'webvision': 50, 'food101n': 101}

def init():
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    save_dir = os.path.join(args.save_dir, args.experiment_name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    args.num_classes = NUM_CLASSES[args.dataset]
    set_logger(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))
    return args


def get_args():
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar-100')
    parser.add_argument('--noise_type', type=str, default='noisy')
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--validation_split', type=float, default=0.1)

    # training config
    parser.add_argument('--alg', type=str, default='Co-teaching')
    parser.add_argument('--experiment_name', type=str, default='debug')
    parser.add_argument('--arc', type=str, default='preact18')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='save/debug')
    parser.add_argument('--n_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLR')
    parser.add_argument('--milestones', type=int, nargs='*', default=[80])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--early_stopping_patient', type=int, default=2000)
    parser.add_argument('--mixup_alpha', type=float, default=0.5)
    parser.add_argument('--refine_weight', type=float, default=0.3)

    # sample selector config
    parser.add_argument('--selector', type=str, default='MT')
    parser.add_argument('--warmup_epoch', type=int, default=30)
    parser.add_argument('--decay', type=float, default=0.9)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--reg_covar', type=float, default=1e-6)
    parser.add_argument('--p_threshold', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.01)

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(args):
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    args = init()
    if args.alg == 'CE':
        pass
    elif args.alg == 'Co-teaching':
        trainer = CoTeaching(args)
        trainer.train()
    elif args.alg == 'CNLCU':
        trainer = CNLCU(args)
        trainer.train()
    # this is the original implementation of AUM
    elif args.alg == 'AUM':
        trainer = AUM(args)
        trainer.train()
    elif args.alg == 'dynamic_sample_selection':
        trainer = DSS(args)
        trainer.train()
    elif args.alg == 'FINE':
        trainer = FINE(args)
        trainer.train()
    elif args.alg == 'HMW':
        trainer = HMW(args)
        trainer.train()
    else:
        logging.info(f'Algorithm {args.alg} is not supported.')

