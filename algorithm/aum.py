import logging
import os
import math
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.dataloaders import load_datasets
from utils import metric
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler
from model.SampleSelector import AUMSelector
from model.model import get_model


class AUM:
    def __init__(self, args):
        self.args = args
        self.loader = load_datasets(args)
        self.train_loader = self.loader.run('train')
        self.valid_loader = self.loader.run('valid')
        self.test_loader = self.loader.run('test')
        if self.loader.train_gt_labels is not None:
            torch.save(self.loader.train_gt_labels, os.path.join(args.save_dir, 'clean_label.pth'))
            self.clean_or_not = self.loader.train_gt_labels == self.loader.train_noisy_labels
            logging.info(f'actual noise rate: {1 - self.clean_or_not.mean()}')
        else:
            self.clean_or_not = None
        torch.save(self.loader.train_noisy_labels, os.path.join(args.save_dir, 'noisy_label.pth'))
        self.labels = torch.LongTensor(self.loader.train_noisy_labels)
        args.n_samples = len(self.labels)
        self.n_samples = len(self.labels)

        self._best_test_acc = 0.0
        self._test_acc = []
        self._best_valid_acc = 0.0
        self._early_stopping_cnt = 0

        self.clean_mask = torch.ones(self.n_samples).bool().to(args.device)

    def add_threshold_samples(self, keep_idx=set()):
        res = self.labels.clone()
        n_threshold_sample = math.ceil(self.n_samples / (self.args.num_classes + 1))
        candidate = list(set(range(self.n_samples)) - keep_idx)
        threshold_idx = random.sample(candidate, n_threshold_sample)
        res[threshold_idx] = self.args.num_classes
        return res, threshold_idx

    def train(self):
        # generate two set of threshold samples
        labels1, threshold_idx1 = self.add_threshold_samples()
        labels2, threshold_idx2 = self.add_threshold_samples(set(threshold_idx1))
        self.args.num_classes = self.args.num_classes + 1

        # First threshold sample run
        model = get_model(self.args).to(self.args.device)
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scaler = GradScaler()
        sample_selector = AUMSelector(self.args, labels1)
        for epoch in range(self.args.milestones[0]):
            self._train_epoch(model, optimizer, scaler, labels1, sample_selector)
        clean_mask1 = sample_selector.select_clean_label(threshold_idx1)

        # Second threshold sample run
        model = get_model(self.args).to(self.args.device)
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scaler = GradScaler()
        sample_selector = AUMSelector(self.args, labels2)
        for epoch in range(self.args.milestones[0]):
            self._train_epoch(model, optimizer, scaler, labels2, sample_selector)
        clean_mask2 = sample_selector.select_clean_label(threshold_idx2)

        # select clean samples
        self.clean_mask = clean_mask1 | clean_mask2
        self._eval_sample_selection()
        torch.save(self.clean_mask, os.path.join(self.args.save_dir, 'selected_clean_label.pth'))

        # # reconstruct train_loader
        # temp = self.clean_mask.cpu().numpy()
        # train_images = self.train_loader.dataset.images[temp]
        # train_labels = self.train_loader.dataset.observed_label[temp]
        # load_func = self.train_loader.dataset.load_func
        # transform = self.train_loader.dataset.transform
        # train_dateset = LNLDataset(train_images, train_labels, load_func, transform)
        # self.train_loader = DataLoader(
        #     dataset=train_dateset,
        #     batch_size=self.args.batch_size,
        #     shuffle=True,
        #     pin_memory=True,
        #     num_workers=8
        # )

        # training loop
        # recover num_classes, re-initialize model, optimizer and scheduler
        self.args.num_classes = self.args.num_classes - 1
        self.model = get_model(self.args).to(self.args.device)
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                         gamma=self.args.gamma)
        for epoch in range(self.args.n_epoch):
            self.args.current_epoch = epoch + 1
            logging.info(f'########## epoch{epoch + 1} ##########')
            logging.info(f'learning rate: {scheduler.get_last_lr()}')
            self._train_epoch(self.model, optimizer, scaler, self.labels)
            scheduler.step()
            # valid
            if self.valid_loader is not None:
                self._valid()
            # test
            self._test()
        self._load_checkpoint('best.pkl')
        self._test()

    def _train_epoch(self, model, optimizer, scaler, labels, sample_selector=None):
        model.train()
        for (image_w, _), _, batch_index in tqdm(self.train_loader):
            image = image_w.to(self.args.device)
            batch_label = labels[batch_index].to(self.args.device)
            batch_index = batch_index.to(self.args.device)
            clean_index = self.clean_mask[batch_index]

            optimizer.zero_grad()
            with torch.autocast(device_type=self.args.device, dtype=torch.float16):
                logit = model(image)
                loss = F.cross_entropy(logit, batch_label, reduction='none')
                loss = torch.where(clean_index, loss, torch.zeros_like(loss)).mean()
                # loss = F.cross_entropy(logit, batch_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if sample_selector is not None:
                sample_selector.update_aum(batch_index, logit.detach().float())

    @torch.no_grad()
    def _valid(self):
        logit, labels = self._infer(self.valid_loader)
        acc, _ = metric(logit, labels)
        if self._best_valid_acc > acc:
            self._early_stopping_cnt += 1
        else:
            self._best_valid_acc = acc
            self._early_stopping_cnt = 0
            self._save_checkpoint('best.pkl')
        logging.info(f'valid_acc: {acc}')
        logging.info(f'early stopping cnt: {self._early_stopping_cnt}')

    @torch.no_grad()
    def _test(self):
        logit, labels = self._infer(self.test_loader)
        acc, _ = metric(logit, labels)
        self._test_acc.append(acc)
        self._best_test_acc = max(acc, self._best_test_acc)
        logging.info(f'test_acc: {acc}')
        logging.info(f'best_test_acc: {self._best_test_acc}')
        logging.info(f'last_test_acc: {sum(self._test_acc[-10:]) / len(self._test_acc[-10:])}')

    @torch.no_grad()
    def _infer(self, loader):
        self.model.eval()
        logit = torch.zeros(len(loader.dataset), self.args.num_classes)
        labels = torch.zeros(len(loader.dataset)).long()
        for image, label, index in loader:
            image = image.to(self.args.device)
            logit[index] = self.model(image).detach().cpu()
            labels[index] = label
        return logit, labels

    def _eval_sample_selection(self):
        clean_pred = self.clean_mask.cpu().numpy()
        clean_true = self.clean_or_not
        p = precision_score(clean_true, clean_pred) * 100
        r = recall_score(clean_true, clean_pred) * 100
        f1 = f1_score(clean_true, clean_pred) * 100
        logging.info(f'selected clean labels: {clean_pred.sum()}')
        logging.info(f'sample selection result w.r.t clean labels: P: {p:.2f}, R: {r:.2f}, f1: {f1:.2f}')

    def _save_checkpoint(self, name):
        checkpoint_path = os.path.join(self.args.save_dir, name)
        logging.info(f'saving checkpoint to {checkpoint_path}...')
        state = {
            'model': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, checkpoint_path)

    def _load_checkpoint(self, name):
        checkpoint_path = os.path.join(self.args.save_dir, name)
        logging.info(f'loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])

