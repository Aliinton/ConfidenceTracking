import os
import json
import random
import argparse
import datetime
import logging
import torch
import torch.nn.functional as F
import numpy as np
from dataloader.dataloaders import *
from model.model import get_model
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from model.SampleSelector import CT

NUM_CLASSES = {'cifar-10': 10, 'cifar-100': 100, 'cifar-10n': 10, 'cifar-100n': 100, 'webvision': 50, 'food101n': 101}


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
    parser.add_argument('--dataset', type=str, default='food101n')
    parser.add_argument('--noise_type', type=str, default='real')
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--validation_split', type=float, default=None)

    # training config
    parser.add_argument('--experiment_name', type=str, default='food101n')    
    parser.add_argument('--alg', type=str, default='CORSE')
    parser.add_argument('--arc', type=str, default='resnet50')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./save/CORSE/')
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLR')
    parser.add_argument('--milestones', type=int, nargs='*', default=[10, 20])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--ramup_length', type=float, default=10)

    # sample selector config
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--ct_refine', action='store_true', default=False)

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


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


@torch.no_grad()
def test(args, net, data):
    net.eval()
    logits, labels = [], []
    for img, label, idx in data:
        img = img.to(args.device)
        logits.append(net(img).cpu())
        labels.append(label)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    acc, top5 = metric(logits, labels)
    return acc, top5


def metric(logit, target, topk=(1, 5)):
    _, pred = logit.topk(max(topk), dim=1, largest=True, sorted=True)
    correct = pred.eq(target.reshape(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(100 * correct_k / target.size(0))
    return res


def warmup(args, net, optimizer, scaler, dataloader, noise_prior):
    net.train()
    for img, label, index in tqdm(dataloader):
        img, label = img.to(args.device), label.to(args.device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = net(img)
            loss_ = -torch.log(output.softmax(1) + 1e-8)
            sup_loss = F.cross_entropy(output, label)
            reg_loss = (noise_prior * loss_).sum(1)
            loss = sup_loss - args.beta * reg_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def linear_rampup(x, current, warmup, rampup_length=10):
    current = np.clip((current-warmup) / rampup_length, 0.0, 1.0)
    return x*float(current)

def train(epoch, args, net, optimizer, scaler, train_loader, sample_selector, noise_prior, clean_mask_CT):
    net.train()
    for (img_w, img_s), labels, idx in tqdm(train_loader):
        img_w, img_s = img_w.to(args.device), img_s.to(args.device)
        labels = labels.to(args.device)
        idx = idx.to(args.device)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logit = net(img_w)
            loss_ = -torch.log(logit.softmax(1) + 1e-8)
            loss = F.cross_entropy(logit, labels, reduction='none')

            # sample selection
            clean_mask_corse = loss.lt(loss_.mean(1))
            sample_selector.update(idx, logit.softmax(1).detach(), epoch)
            if epoch < args.warmup_epoch:
                batch_clean_mask = torch.ones_like(loss).bool()
            else:
                if args.ct_refine:
                    batch_clean_mask = clean_mask_CT[idx] | clean_mask_corse
                else:
                    batch_clean_mask = clean_mask_corse
            clean_mask[idx] = batch_clean_mask

            # cal loss
            reg_loss = (noise_prior * loss_).sum(1)
            if args.dataset == 'webvision':
                beta = linear_rampup(args.beta, epoch, args.warmup_epoch, args.ramup_length)
            elif args.dataset == 'food101n':
                if epoch < args.warmup_epoch:
                    beta = 0.0
                else:
                    beta = args.beta
            loss = loss - beta * reg_loss
            loss = torch.where(batch_clean_mask, loss, torch.zeros_like(loss)).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def save_checkpoint(args, net, name):
    torch.save({
        'net': net.state_dict(),
    }, os.path.join(args.save_dir, name))


def load_checkpoint(args, net, name):
    checkpoint = torch.load(os.path.join(args.save_dir, name))
    net.load_state_dict(checkpoint['net'])


if __name__ == '__main__':
    args = init()

    test_log = []
    auc_log = []
    best_test_top1 = 0
    best_valid_top1 = 0.0
    valid_top1_log, valid_top5_log = [], []
    test_top1_log, test_top5_log = [], []

    # define dataloader, model, optimizer, scheduler
    loader = load_datasets(args)
    args.n_samples = len(loader.train_noisy_labels)
    net = get_model(args).to(args.device)
    optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # define sample selector
    train_labels = torch.LongTensor(loader.train_noisy_labels).to(args.device)
    clean_mask = torch.ones(len(train_labels)).bool().to(args.device)
    sample_selector = CT(args, train_labels)

    # training
    train_loader = loader.run('train')
    valid_loader = loader.run('valid')
    test_loader = loader.run('test')
    for epoch in range(args.n_epoch):
        logging.info(f'####### epoch{epoch + 1} #######')
        logging.info(f'|selected|/|all|: {clean_mask.sum().cpu().item()}/{len(clean_mask)}')
        # update noise prior
        select_label = train_labels[clean_mask]
        noise_prior = torch.bincount(select_label, minlength=args.num_classes)
        noise_prior = noise_prior / noise_prior.sum()
        clean_mask_CT = sample_selector.select_clean_labels(epoch - 1)
        train(epoch, args, net, optimizer, scaler, train_loader, sample_selector, noise_prior, clean_mask_CT)
        scheduler.step()

        # test
        if valid_loader is not None:
            valid_top1, valid_top5 = test(args, net, valid_loader)
            logging.info(f'Valid top1: {valid_top1}, top5: {valid_top5}')
            if valid_top1 > best_valid_top1:
                logging.info(f'New best valid top1, save checkpoint to {os.path.join(args.save_dir, "best_valid.pth")}')
                best_valid_top1 = valid_top1
                save_checkpoint(args, net, 'best_valid.pth')
            valid_top1_log.append(valid_top1)
            valid_top5_log.append(valid_top5)
            logging.info(f'Avg valid top1: {sum(valid_top1_log[-10:])/len(valid_top1_log[-10:])}, '
                         f'Avg valid top5: {sum(valid_top5_log[-10:])/len(valid_top5_log[-10:])}')

        test_top1, test_top5 = test(args, net, test_loader)
        test_top1_log.append(test_top1)
        test_top5_log.append(test_top5)
        if test_top1 > best_test_top1:
            logging.info(f'New best test top1, save checkpoint to {os.path.join(args.save_dir, "best_test.pth")}')
            best_test_top1 = test_top1
            save_checkpoint(args, net, 'best_test.pth')
        logging.info(f'test top1: {test_top1}, top5: {test_top5}')
        logging.info(f'Avg test top1: {sum(test_top1_log[-10:]) / len(test_top1_log[-10:])}, '
                     f'Avg test top5: {sum(test_top5_log[-10:]) / len(test_top5_log[-10:])}')
        save_checkpoint(args, net, 'last.pth')


    save_checkpoint(args, net, 'last.pth')

    # test
    logging.info(f'Load checkpoint from {os.path.join(args.save_dir, "best_valid.pth")}...')
    load_checkpoint(args, net, 'best_valid.pth')
    valid_top1, valid_top5 = test(args, net, valid_loader)
    logging.info(f'Best valid top1: {valid_top1}, top5: {valid_top5}')

    logging.info(f'Load checkpoint from {os.path.join(args.save_dir, "best_test.pth")}...')
    load_checkpoint(args, net, 'best_test.pth')
    test_top1, test_top5 = test(args, net, test_loader)
    logging.info(f'Best Test top1: {test_top1}, top5: {test_top5}')
