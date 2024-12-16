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
from model.SampleSelector import CT, DIST
from sklearn.metrics import precision_score, recall_score, f1_score

NUM_CLASSES = {'cifar-10': 10, 'cifar-100': 100, 'cifar-10n': 10, 'cifar-100n': 100, 'webvision': 50, 'food101n': 101}

CELoss = torch.nn.CrossEntropyLoss()


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
    parser.add_argument('--alg', type=str, default='DISC')
    parser.add_argument('--experiment_name', type=str, default='food101n')
    parser.add_argument('--arc', type=str, default='resnet50')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='save/DISC')
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLR')
    parser.add_argument('--milestones', type=int, nargs='*', default=[10, 20])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # sample selector config
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--ct_refine', action='store_true', default=False)

    # MixUp config
    parser.add_argument('--mixup_alpha', type=float, default=5.0)

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


def conf_penalty(outputs):
    probs = torch.softmax(outputs, dim=1)
    return torch.mean(torch.sum(probs.log() * probs, dim=1))


def warmup(args, net, optimizer, scaler, dataloader, sample_selector_w, sample_selector_s):
    net.train()
    for (img_w, img_s), label, index in tqdm(dataloader):
        img_w, img_s, label = img_w.to(args.device), img_s.to(args.device), label.to(args.device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            all_inputs = torch.cat([img_w, img_s], dim=0)
            output = net(all_inputs)
            w_logit, s_logit = torch.chunk(output, 2, dim=0)
            loss = F.cross_entropy(w_logit, label) + F.cross_entropy(s_logit, label)

            sample_selector_w.update(index, w_logit.softmax(1).detach(), args.epoch)
            sample_selector_s.update(index, s_logit.softmax(1).detach(), args.epoch)
            if args.ct_refine:
                sample_selector_ct_w.update(index, w_logit.softmax(1).detach(), args.epoch)
                sample_selector_ct_s.update(index, s_logit.softmax(1).detach(), args.epoch)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def gce_loss(logits, labels, q=0.7):
    labels = F.one_hot(labels, num_classes=logits.shape[-1])
    probs = F.softmax(logits, dim=-1)
    loss = (1 - torch.pow(torch.sum(labels * probs, dim=-1), q)) / q
    return loss


def mixup_loss(args, net, img, target):
    n = len(target)
    target = F.one_hot(target, num_classes=args.num_classes)
    l = np.random.beta(args.mixup_alpha, args.mixup_alpha)
    l = max(l, 1 - l)
    temp = torch.randperm(n)
    img_a, img_b = img, img[temp]
    target_a, target_b = target, target[temp]
    mixed_img = l * img_a + (1 - l) * img_b
    mixed_target = l * target_a + (1 - l) * target_b
    loss = F.cross_entropy(net(mixed_img), mixed_target)
    return loss



def train(args, net, optimizer, scaler, dataloader,
          clean_mask, hard_mask, mix_mask, weak_label, sample_selector_w, sample_selector_s):
    log = []
    net.train()
    for (image_w, image_s), label, index in tqdm(dataloader):
        image_w = image_w.to(args.device)
        image_s = image_s.to(args.device)
        label = label.to(args.device)
        index = index.to(args.device)
        batch_clean_mask = clean_mask[index]
        batch_hard_mask = hard_mask[index]
        batch_mix_mask = mix_mask[index]
        batch_weak_label = weak_label[index]

        optimizer.zero_grad()
        with (torch.autocast(device_type='cuda', dtype=torch.float16)):
            img = torch.cat([image_w, image_s], dim=0)
            logit_w, logit_s = net(img).chunk(2)

            Lc = F.cross_entropy(logit_w, label, reduction='none') + F.cross_entropy(logit_s, label, reduction='none')
            Lc = torch.where(batch_clean_mask, Lc, torch.zeros_like(Lc)).mean()

            Lh = gce_loss(logit_w, label) + gce_loss(logit_s, label)
            Lh = torch.where(batch_hard_mask, Lh, torch.zeros_like(Lh)).mean()

            if batch_mix_mask.sum() > 0:
                Lm = mixup_loss(args, net, image_w[batch_mix_mask], batch_weak_label[batch_mix_mask])
                Lm += mixup_loss(args, net, image_s[batch_mix_mask], batch_weak_label[batch_mix_mask])
                Lm *= batch_mix_mask.sum() / len(batch_mix_mask)
            else:
                Lm = 0

            loss = Lc + Lh + Lm

            log.append({
                'Lc': Lc.detach().item(),
                'Lh': Lh.detach().item(),
                'Lm': 0 if isinstance(Lm, int) else Lm.detach().item()
            })

            pred_w, pred_s = logit_w.softmax(-1).detach(), logit_s.softmax(-1).detach()
            sample_selector_w.update(index, pred_w, args.epoch)
            sample_selector_s.update(index, pred_s, args.epoch)
            if args.ct_refine:
                sample_selector_ct_w.update(index, pred_w, args.epoch)
                sample_selector_ct_s.update(index, pred_s, args.epoch)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for key in log[0]:
        logging.info(f'{key}: {sum([_[key] for _ in log]) / len(log)}')


def eval_split(mask, clean_or_not, split):
    p = precision_score(clean_or_not, mask) * 100
    r = recall_score(clean_or_not, mask) * 100
    f = f1_score(clean_or_not, mask)
    logging.info(f'{split}: Precision: {p:.2f}, Recall: {r:.2f}, F1-score: {f:.2f}')


def save_checkpoint(args, net, name):
    torch.save({
        'net': net.state_dict(),
    }, os.path.join(args.save_dir, name))


def load_checkpoint(args, net, name):
    checkpoint = torch.load(os.path.join(args.save_dir, name))
    net.load_state_dict(checkpoint['net'])


if __name__ == '__main__':
    args = init()

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
    if loader.train_gt_labels is not None:
        clean_or_not = loader.train_gt_labels == loader.train_noisy_labels
        gt_labels = torch.LongTensor(loader.train_gt_labels).to(args.device)
    else:
        clean_or_not = None
    if args.ct_refine:
        sample_selector_w = DIST(args, train_labels)
        sample_selector_s = DIST(args, train_labels)
        sample_selector_ct_w = CT(args, train_labels)
        sample_selector_ct_s = CT(args, train_labels)
    else:
        sample_selector_w = DIST(args, train_labels)
        sample_selector_s = DIST(args, train_labels)

    # training
    best_valid_top1 = 0.0
    train_loader = loader.run('train')
    valid_loader = loader.run('valid')
    test_loader = loader.run('test')
    clean_mask = torch.ones(args.n_samples).bool().to(args.device)
    hard_mask = torch.zeros(args.n_samples).bool().to(args.device)
    mix_mask = torch.zeros(args.n_samples).bool().to(args.device)
    weak_label = torch.zeros(args.n_samples).to(args.device)
    for epoch in range(args.n_epoch):
        logging.info(f'####### epoch{epoch + 1} #######')
        logging.info(f'learning rate: {scheduler.get_last_lr()}')
        args.epoch = epoch
        if epoch < args.warmup_epoch:
            warmup(args, net, optimizer, scaler, train_loader, sample_selector_w, sample_selector_s)
        else:
            train(args, net, optimizer, scaler, train_loader,
                  clean_mask, hard_mask, mix_mask, weak_label, sample_selector_w, sample_selector_s)
        scheduler.step()

        # sample selection
        clean_w = sample_selector_w.select_clean_labels(epoch)
        clean_s = sample_selector_s.select_clean_labels(epoch)

        # split clean and hard
        if args.ct_refine:
            clean_ct_w = sample_selector_ct_w.select_clean_labels(epoch)
            clean_ct_s = sample_selector_ct_s.select_clean_labels(epoch)
            logging.info(f'ct_weak_view: {clean_ct_w.sum()}, ct_strong_view: {clean_ct_s.sum()}')
            clean_mask = (clean_w & clean_s) | (clean_ct_w | clean_ct_s)
            hard_mask = (clean_w | clean_s) & (~clean_mask)
        else:
            clean_mask = (clean_w & clean_s)
            hard_mask = (clean_w | clean_s) & (~clean_mask)

        # select purified samples
        avg_pred = (sample_selector_w.pred + sample_selector_s.pred) / 2
        confidence, pseudo_label = avg_pred.max(1)
        purify_threshold = (sample_selector_w.threshold + sample_selector_s.threshold) / 2 + args.sigma
        purify_threshold = purify_threshold.clamp(max=0.99)
        purified_mask = confidence.gt(purify_threshold) & (~(clean_mask | hard_mask))

        # split mixed set
        weak_label = torch.where(purified_mask, pseudo_label, train_labels)
        mix_mask = purified_mask | clean_mask | hard_mask

        # logging
        logging.info(f'|C|: {clean_mask.sum()}, |H|: {hard_mask.sum()}, |P|: {purified_mask.sum()}')
        if clean_or_not is not None:
            eval_split(clean_mask.detach().cpu().numpy(), clean_or_not, 'clean')
            eval_split(hard_mask.detach().cpu().numpy(), clean_or_not, 'hard')
            weak_label_acc = weak_label.eq(gt_labels)[mix_mask].float().mean().cpu().item()
            logging.info(f'Weak label accuracy: {weak_label_acc}')


        # test
        if valid_loader is not None:
            valid_top1, valid_top5 = test(args, net, valid_loader)
            logging.info(f'Valid top1: {valid_top1}, top5: {valid_top5}')
            valid_top1_log.append(valid_top1)
            valid_top5_log.append(valid_top5)
            logging.info(f'Avg valid top1: {sum(valid_top1_log[-10:])/len(valid_top1_log[-10:])}, '
                         f'Avg valid top5: {sum(valid_top5_log[-10:])/len(valid_top5_log[-10:])}')

        test_top1, test_top5 = test(args, net, test_loader)
        logging.info(f'test top1: {test_top1}, top5: {test_top5}')
        test_top1_log.append(test_top1)
        test_top5_log.append(test_top5)
        logging.info(f'Avg test top1: {sum(test_top1_log[-10:]) / len(test_top1_log[-10:])}, '
                     f'Avg test top5: {sum(test_top5_log[-10:]) / len(test_top5_log[-10:])}, '
                     f'Best test top1: {max(test_top1_log)}, Best test top5: {max(test_top5_log)}')
        save_checkpoint(args, net, 'last.pth')


