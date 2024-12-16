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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from dataloader.dataloaders import *
from L2D.resnet import ResNet
from sklearn.model_selection import train_test_split
from model.SampleSelector import L2DSelector


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
    parser.add_argument('--noise_rate', type=float, default=0.3)
    parser.add_argument('--validation_split', type=float, default=None)

    # training config
    parser.add_argument('--experiment_name', type=str, default='train_noise_detector')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save_dir', type=str, default='./save/L2D')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLR')
    parser.add_argument('--milestones', type=int, nargs='*', default=[100, 150])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

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


def generate_train_dynamic(args, model, optimizer, scheduler, train_loader, test_loader):
    scaler = GradScaler()
    res = torch.Tensor(args.n_samples, args.n_epoch)
    for e in tqdm(range(args.n_epoch)):
        # train
        model.train()
        for (image_w, _), label, index in train_loader:
            image = image_w.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logit = model(image)
                loss = F.cross_entropy(logit, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # collect training dynamics
        model.eval()
        for (image_w, _), label, index in train_loader:
            image = image_w.to(args.device)
            label = label.to(args.device)
            logit = model(image)
            res[index, e] = torch.gather(logit.softmax(-1).float(), dim=1, index=label.unsqueeze(1)).detach().cpu().squeeze(1)

        # test
        if e % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            for image, label, _ in test_loader:
                image = image.to(args.device)
                label = label.to(args.device)
                pred = model(image).argmax(1)
                correct += label.eq(pred).sum().detach().item()
                total += len(label)
            logging.info(f'Epoch[{e}/200] Test Acc:{(correct/total)*100:.2f}% Learning Rate: {scheduler.get_last_lr()}')

    return res



if __name__ == '__main__':
    args = init()
    loader = load_datasets(args)
    args.n_samples = len(loader.train_noisy_labels)
    logging.info(args)
    clean_or_not = torch.LongTensor(loader.train_gt_labels == loader.train_noisy_labels)
    logging.info(f'actual noise rate: {1 - clean_or_not.float().mean()}')
    train_loader = loader.run('train')
    test_loader = loader.run('test')
    model = ResNet(num_classes=args.num_classes).to(args.device)
    optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # collect training dynamics
    logging.info('############# collect training dynamics #############')
    training_dynamics = generate_train_dynamic(args, model, optimizer, scheduler, train_loader, test_loader)

    # split training and test dataset
    train_x, test_x, train_y, test_y = train_test_split(training_dynamics, clean_or_not, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # train noise detector
    noise_detector = L2DSelector().to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(noise_detector.parameters(), lr=0.1)
    max_epoch = 10
    best_acc = 0

    for epoch in range(max_epoch):
        noise_detector.train()
        loss_sigma = 0.0
        correct = 0.0
        total = 0.0
        for i, (train_data, train_label) in enumerate(train_dataloader):
            train_data = train_data.to(args.device).unsqueeze(-1)
            train_label = train_label.to(args.device)
            out = noise_detector(train_data)
            loss = criterion(out, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out.data, 1)
            total += train_label.size(0)
            correct += (predicted == train_label).squeeze().sum().cpu().numpy()
            loss_sigma += loss.item()

        logging.info("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, max_epoch, loss_sigma, correct / total))

        # evaluation on test set
        noise_detector.eval()
        conf_matrix = np.zeros((2, 2))
        with torch.no_grad():
            for it, (test_data, test_label) in enumerate(test_dataloader):
                test_data = test_data.to(args.device).unsqueeze(-1)
                test_label = test_label.to(args.device)
                test_out = noise_detector(test_data)

                _, predicted = torch.max(test_out.data, 1)
                for i in range(predicted.shape[0]):
                    conf_matrix[test_label[i], predicted[i]] += 1

        test_acc = np.diag(conf_matrix).sum() / np.sum(conf_matrix)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(noise_detector.state_dict(), os.path.join(f'./save/L2D/pretrained_noise_detector_{args.dataset}.pth'))


