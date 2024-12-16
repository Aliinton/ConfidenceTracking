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
from sklearn.mixture import GaussianMixture
from model.SampleSelector import CT

NUM_CLASSES = {'cifar-10': 10, 'cifar-100': 100, 'cifar-10n': 10, 'cifar-100n': 100, 'clothing1m': 14, 'webvision': 50, 'food101n': 101}

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
    parser.add_argument('--experiment_name', type=str, default='food101n')
    parser.add_argument('--alg', type=str, default='DivideMix')
    parser.add_argument('--arc', type=str, default='resnet50')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='save/devidemix')
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_batches', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLR')
    parser.add_argument('--milestones', type=int, nargs='*', default=[10, 20])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--lambda_u', default=0, type=float)

    # sample selector config
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--ct_refine', action='store_true', default=False)
    parser.add_argument('--p_threshold', type=float, default=0.5)
    parser.add_argument('--reg_covar', type=float, default=5e-4)

    # MixUp config
    parser.add_argument('--mixup_alpha', type=float, default=0.5)

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


def test2(args, net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, idx in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))


@torch.no_grad()
def test(args, net1, net2, data):
    net1.eval()
    net2.eval()
    logits, labels = [], []
    for img, label, idx in data:
        img = img.to(args.device)
        logits.append((net1(img) + net2(img)).cpu())
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


def warmup(args, net, optimizer, scaler, dataloader):
    net.train()
    for img, label, index in tqdm(dataloader):
        img, label = img.to(args.device), label.to(args.device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = net(img)
            loss = F.cross_entropy(output, label)
            if args.dataset == 'clothing1m':
                loss += conf_penalty(output)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def train(args, net1, net2, optimizer, scaler, labeled_loader, unlabeled_loader, clean_prob):
    assert len(clean_prob) == len(labeled_loader.dataset)
    # fix one network and train the other
    net1.train()
    net2.eval()

    unlabeled_iter = iter(unlabeled_loader)
    num_iter = (len(labeled_loader.dataset) // args.batch_size) + 1
    for batch_idx, ((img_x1, img_x2), labels_x, idx) in tqdm(enumerate(labeled_loader)):
        try:
            (img_u1, img_u2), _, _ = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            (img_u1, img_u2), _, _ = next(unlabeled_iter)

        batch_size = img_x1.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1, 1), 1)

        w_x = clean_prob[idx].view(-1, 1).to(args.device)
        img_x1, img_x2, labels_x = img_x1.to(args.device), img_x2.to(args.device), labels_x.to(args.device)
        img_u1, img_u2 = img_u1.to(args.device), img_u2.to(args.device)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net1(img_u1)
                outputs_u12 = net1(img_u2)
                outputs_u21 = net2(img_u1)
                outputs_u22 = net2(img_u2)

                pu = (outputs_u11.softmax(1) + outputs_u12.softmax(1) + outputs_u21.softmax(1) + outputs_u22.softmax(
                    1)) / 4
                ptu = pu ** (1 / args.T)

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x1 = net1(img_x1)
                outputs_x2 = net1(img_x2)

                px = (outputs_x1.softmax(1) + outputs_x2.softmax(1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / args.T)

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)
                targets_x = targets_x.detach()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # mixmatch
            l = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([img_x1, img_x2, img_u1, img_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net1(mixed_input)
            logits_x = logits[:batch_size * 2]
            logits_u = logits[batch_size * 2:]
            Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], dim=1))
            Lu = torch.mean((logits_u.softmax(1) - mixed_target[batch_size * 2:]) ** 2)
            lambda_u = linear_rampup(args.lambda_u, epoch + batch_idx / num_iter, args.warmup_epoch)

            prior = torch.ones(args.num_classes) / args.num_classes
            prior = prior.to(args.device)
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + penalty + lambda_u * Lu

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def linear_rampup(x, current, warmup, rampup_length=16):
    current = np.clip((current - warmup) / rampup_length, 0.0, 1.0)
    return x * float(current)


def eval_train(epoch, args, net, eval_loader, sample_selector, loader):
    net.eval()
    loss = torch.zeros(len(eval_loader.dataset))
    logits = torch.zeros(len(eval_loader.dataset), args.num_classes).to(args.device)
    with torch.no_grad():
        for img, labels, idx in tqdm(eval_loader):
            img, labels = img.to(args.device), labels.to(args.device)
            outputs = net(img)
            logits[idx] = outputs
            loss[idx] = F.cross_entropy(outputs, labels, reduction='none').cpu()

        sample_selector.update(torch.arange(len(eval_loader.dataset)), logits.softmax(1).detach(), epoch)


    if args.dataset == 'clothing1m':
        idx = loader.sample_balanced_subset(args.num_batches * args.batch_size)
    else:
        idx = torch.arange(len(loss))

    temp = loss[idx].clone()
    v_min = temp.min()
    v_max = temp.max()
    temp = (temp - v_min) / (v_max - v_min)
    loss = (loss - v_min) / (v_max - v_min)

    # fit a two-component GMM to the loss
    temp = temp.reshape(-1, 1).numpy()
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=args.reg_covar)
    gmm.fit(temp)
    prob = gmm.predict_proba(loss.reshape(-1, 1).numpy())
    prob = prob[:, gmm.means_.argmin()]
    prob = torch.Tensor(prob)
    if args.ct_refine:
        res = sample_selector.select_clean_labels(epoch).cpu()
        add_mask = res & prob.lt(args.p_threshold)
        logging.info(f'Select: {res.sum()}, Add {add_mask.sum()}')
        prob[res] = 1
    return prob, idx


def save_checkpoint(args, net1, net2, name):
    torch.save({
        'net1': net1.state_dict(),
        'net2': net2.state_dict(),
    }, os.path.join(args.save_dir, name))


def load_checkpoint(args, net1, net2, name):
    checkpoint = torch.load(os.path.join(args.save_dir, name))
    net1.load_state_dict(checkpoint['net1'])
    net2.load_state_dict(checkpoint['net2'])


if __name__ == '__main__':
    args = init()

    valid_top1_log, valid_top5_log = [], []
    test_top1_log, test_top5_log = [], []
    best_valid_top1 = 0.0

    # define dataloader, model, optimizer, scheduler
    loader = load_datasets(args)
    args.n_samples = len(loader.train_noisy_labels)
    net1 = get_model(args).to(args.device)
    net2 = get_model(args).to(args.device)
    optimizer1 = torch.optim.SGD(params=filter(lambda p: p.requires_grad, net1.parameters()),
                                 lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    optimizer2 = torch.optim.SGD(params=filter(lambda p: p.requires_grad, net2.parameters()),
                                 lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    scaler = GradScaler()
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=args.milestones, gamma=args.gamma)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=args.milestones, gamma=args.gamma)

    # define sample selector
    train_labels = torch.LongTensor(loader.train_noisy_labels)
    clean_true = loader.train_gt_labels == loader.train_noisy_labels
    sample_selector1 = CT(args, train_labels)
    sample_selector2 = CT(args, train_labels)

    # training
    eval_train_loader = loader.run('eval_train')
    valid_loader = loader.run('valid')
    test_loader = loader.run('test')
    clean_mask1 = torch.ones(args.n_samples).bool()
    clean_mask2 = torch.ones(args.n_samples).bool()
    for epoch in range(args.n_epoch):
        logging.info(f'####### epoch{epoch + 1} #######')
        args.epoch = epoch
        # warmup
        if epoch < args.warmup_epoch:
            # warmup net1
            warmup_loader = loader.run('warmup', num_batches=args.num_batches)
            warmup(args, net1, optimizer1, scaler, warmup_loader)
            scheduler1.step()

            # warmup net2
            warmup_loader = loader.run('warmup', num_batches=args.num_batches)
            warmup(args, net2, optimizer2, scaler, warmup_loader)
            scheduler2.step()

        else:
            # update clean mask
            clean_mask1 = clean_prob1.gt(args.p_threshold)
            clean_mask2 = clean_prob2.gt(args.p_threshold)

            # Train Net1
            labeled_loader, unlabeled_loader = loader.run('ssl', clean_mask=clean_mask2, train_idx=idx2)
            train(args, net1, net2, optimizer1, scaler, labeled_loader, unlabeled_loader, clean_prob2[idx2][clean_mask2[idx2]])
            scheduler1.step()

            # Train Net2
            labeled_loader, unlabeled_loader = loader.run('ssl', clean_mask=clean_mask1, train_idx=idx1)
            train(args, net2, net1, optimizer2, scaler, labeled_loader, unlabeled_loader, clean_prob1[idx1][clean_mask1[idx1]])
            scheduler2.step()

        # valid
        if valid_loader is not None:
            valid_top1, valid_top5 = test(args, net1, net2, valid_loader)
            logging.info(f'Valid top1: {valid_top1}, top5: {valid_top5}')
            if valid_top1 > best_valid_top1:
                best_valid_top1 = valid_top1
                save_checkpoint(args, net1, net2, 'best.pth')
                logging.info(f'New best valid top1, save checkpoint to {os.path.join(args.save_dir, "best.pth")}')
            valid_top1_log.append(valid_top1)
            valid_top5_log.append(valid_top5)
            logging.info(f'Avg valid top1: {sum(valid_top1_log[-10:])/len(valid_top1_log[-10:])}, '
                         f'Avg valid top5: {sum(valid_top5_log[-10:])/len(valid_top5_log[-10:])}')
        
        # test
        test_top1, test_top5 = test(args, net1, net2, test_loader)
        logging.info(f'test top1: {test_top1}, top5: {test_top5}')
        test_top1_log.append(test_top1)
        test_top5_log.append(test_top5)
        logging.info(f'Avg test top1: {sum(test_top1_log[-10:]) / len(test_top1_log[-10:])}, '
                     f'Avg test top5: {sum(test_top5_log[-10:]) / len(test_top5_log[-10:])}')
        save_checkpoint(args, net1, net2, 'last.pth')

        # collect model prediction on train data
        clean_prob1, idx1 = eval_train(epoch, args, net1, eval_train_loader, sample_selector1, loader)
        clean_prob2, idx2 = eval_train(epoch, args, net2, eval_train_loader, sample_selector2, loader)

    # test
    logging.info(f'Load checkpoint from {os.path.join(args.save_dir, "best.pth")}...')
    load_checkpoint(args, net1, net2, 'best.pth')
    valid_top1, valid_top5 = test(args, net1, net2, valid_loader)
    logging.info(f'Valid top1: {valid_top1}, top5: {valid_top5}')
    test_top1, test_top5 = test(args, net1, net2, test_loader)
    logging.info(f'Test top1: {test_top1}, top5: {test_top5}')

