import os
import sys
sys.path.append('.')
import json
import random
import argparse
import datetime
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader.dataloaders import *
from model.model import get_model
from L2D.trainer import Trainer


NUM_CLASSES={'cifar-10': 10, 'cifar-100': 100, 'cifar-10n': 10, 'cifar-100n': 100, 'clothing1m': 14, 'webvision': 50}

def init():
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    save_dir = os.path.join(args.save_dir, args.experiment_name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.num_classes = NUM_CLASSES[args.dataset]
    set_logger(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))
    return args


def get_args():
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument('--data_dir', type=str, default='path_to_data')
    parser.add_argument('--dataset', type=str, default='cifar-100')
    parser.add_argument('--noise_type', type=str, default='symmetric')
    parser.add_argument('--noise_rate', type=float, default=0.8)
    parser.add_argument('--validation_split', type=float, default=0.1)

    # training config
    parser.add_argument('--experiment_name', type=str, default='DEBUG')
    parser.add_argument('--arc', type=str, default='preact18')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--n_epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLR')
    parser.add_argument('--milestones', type=int, nargs='*', default=[40, 80])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--noise_detector_path', type=str, default=None)
    parser.add_argument('--delta', type=float, default=0.0)

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
    loader = load_datasets(args)
    torch.save(loader.train_noisy_labels, os.path.join(args.save_dir, 'noisy_label.pth'))
    args.n_samples = len(loader.train_noisy_labels)
    args.noise_detector_path = f'./save/L2D/pretrained_noise_detector_{args.dataset}.pth'
    logging.info(args)
    if loader.train_gt_labels is not None:
        torch.save(loader.train_gt_labels, os.path.join(args.save_dir, 'clean_label.pth'))
        args.clean_or_not = loader.train_gt_labels == loader.train_noisy_labels
        logging.info(f'actual noise rate: {1 - args.clean_or_not.mean()}')
    else:
        args.clean_or_not = None
    train_loader = loader.run('train')
    if args.validation_split is not None:
        valid_loader = loader.run('valid')
    else:
        valid_loader = None
    test_loader = loader.run('test')
    model = get_model(args).to(args.device)
    optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    trainer = Trainer(args, model, optimizer, scheduler, train_loader, valid_loader, test_loader)
    trainer.train()

