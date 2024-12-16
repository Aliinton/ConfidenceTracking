import logging
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from L2D.resnet import ResNet
from utils import metric
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler
from model.SampleSelector import L2DSelector
from dataloader.lnl_dataset import LNLDataset
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, args, model, optimizer, scheduler, train_loader, valid_loader, test_loader):
        self.args = args
        self.noise_rate = args.noise_rate
        self.delta = args.delta
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()

        self._best_test_acc = 0.0
        self._best_valid_acc = 0.0
        self._early_stopping_cnt = 0

        self.sample_selector = L2DSelector().to(args.device)
        self.sample_selector.load_state_dict(torch.load(args.noise_detector_path))
        self.training_dynamics = torch.zeros(args.n_samples, 200).to(args.device)
        self.clean_mask = torch.ones(args.n_samples).bool().to(args.device)

    def train(self):
        # identify noisy labels
        model = ResNet(num_classes=self.args.num_classes).to(self.args.device)
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        for epoch in tqdm(range(200)):
            # training
            model.train()
            for (image_w, _), label, index in self.train_loader:
                image = image_w.to(self.args.device)
                label = label.to(self.args.device)

                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logit = model(image)
                    loss = F.cross_entropy(logit, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            scheduler.step()

            # collecting training dynamics
            model.eval()
            for (image_w, _), label, index in self.train_loader:
                image = image_w.to(self.args.device)
                label = label.to(self.args.device)
                index = index.to(self.args.device)
                output = model(image).softmax(-1).float().detach()
                self.training_dynamics[index, epoch] = torch.gather(output, dim=1, index=label.unsqueeze(1)).squeeze(1)

        self.clean_mask = self.select_clean_sample()
        self._eval_sample_selection()
        torch.save(self.clean_mask, os.path.join(self.args.save_dir, 'selected_clean_label.pth'))

        # training loop
        # Here you can also update the dataloader, and remove the selected noisy data from the dataset.
        # But this usually leads to a slight performance drop.
        # So here we assign 0 weight to selected noisy data, which is consistent with our proposed method.
        for epoch in range(self.args.n_epoch):
            self.args.current_epoch = epoch + 1
            logging.info(f'########## epoch{epoch + 1} ##########')
            logging.info(f'learning rate: {self.scheduler.get_last_lr()}')
            self._train_epoch(epoch, collecting_training_dynamics=False)
            self.scheduler.step()
            # valid
            if self.valid_loader is not None:
                self._valid()
            # test
            self._test()
        self._load_checkpoint('best.pkl')
        self._test()

    def _train_epoch(self, epoch, collecting_training_dynamics=False):
        self.model.train()
        for (image_w, _), label, index in tqdm(self.train_loader):
            image = image_w.to(self.args.device)
            label = label.to(self.args.device)
            index = index.to(self.args.device)
            clean_index = self.clean_mask[index]

            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.args.device, dtype=torch.float16):
                logit = self.model(image)
                if collecting_training_dynamics:
                    self.training_dynamics[index, epoch] = torch.gather(logit.softmax(-1).float(), dim=1,
                                                                        index=label.unsqueeze(1)).squeeze(1)
                    loss = F.cross_entropy(logit, label)
                else:
                    loss = F.cross_entropy(logit, label, reduction='none')
                    loss = torch.where(clean_index, loss, torch.zeros_like(loss)).mean()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
        self._best_test_acc = max(acc, self._best_test_acc)
        logging.info(f'test_acc: {acc}')
        logging.info(f'best_test_acc: {self._best_test_acc}')

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
        clean_true = self.args.clean_or_not
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
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, checkpoint_path)

    def _load_checkpoint(self, name):
        checkpoint_path = os.path.join(self.args.save_dir, name)
        logging.info(f'loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def select_clean_sample(self):
        res = torch.zeros(self.args.n_samples).bool().to(self.args.device)
        clean_prob = torch.zeros(self.args.n_samples).to(self.args.device)
        for idx, td in enumerate(self.training_dynamics):
            clean_prob[idx] = self.sample_selector(td.view(1, -1, 1)).data.softmax(1)[:, 1]
        sorted_idx = torch.argsort(clean_prob, descending=True)
        clean_num = int(self.args.n_samples * (1 - self.noise_rate - self.delta))
        res[sorted_idx[:clean_num]] = True
        return res
