import logging
import os
import sys

import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from dataloader.dataloaders import load_datasets
from model.model import get_model
from utils import metric
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler
from model.SampleSelector import CT
import numpy as np


class FINE:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.n_epoch = args.n_epoch
        self.num_classes = args.num_classes
        self.warmup_epoch = args.warmup_epoch
        self.p_threshold = args.p_threshold

        # dataset
        self.loader = load_datasets(args)
        self.train_loader = self.loader.run('train')
        self.eval_train_loader = self.loader.run('eval_train')
        self.valid_loader = self.loader.run('valid')
        self.test_loader = self.loader.run('test')
        if self.loader.train_gt_labels is not None:
            torch.save(self.loader.train_gt_labels, os.path.join(args.save_dir, 'clean_label.pth'))
            self.clean_or_not = self.loader.train_gt_labels == self.loader.train_noisy_labels
            args.clean_or_not = self.clean_or_not
            logging.info(f'actual noise rate: {1 - self.clean_or_not.mean()}')
        else:
            self.clean_or_not = None
        torch.save(self.loader.train_noisy_labels, os.path.join(args.save_dir, 'noisy_label.pth'))
        self.labels = torch.LongTensor(self.loader.train_noisy_labels)
        args.n_samples = len(self.labels)
        self.n_samples = len(self.labels)

        # model, optimizer, schedule
        self.model = get_model(args).to(args.device)
        self.optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=args.milestones,
                                                              gamma=args.gamma)
        self.scaler = GradScaler()

        # sample selector
        if args.selector == 'FINE+CT':
            self._ct_refine = True
            self.sample_selector = CT(args, self.labels)
            self._ct_clean_mask = torch.ones(self.n_samples).bool().to(args.device)
        else:
            self._ct_refine = False
        self._fine_clean_mask = torch.ones(self.n_samples).bool().to(args.device)
        self._clean_mask = torch.ones(self.n_samples).bool().to(args.device)

        # metrics
        self._best_test_acc = 0.0
        self._test_acc = []
        self._best_valid_acc = 0.0
        self._early_stopping_cnt = 0

    def train(self):
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            logging.info(f'########## epoch{epoch + 1} ##########')
            logging.info(f'learning rate: {self.scheduler.get_last_lr()}')

            self._train_epoch()

            # sample selection
            if (epoch + 1) >= self.warmup_epoch:
                self._sample_selection()

            # valid
            if self.valid_loader is not None:
                self._valid()
                if self._early_stopping_cnt > self.args.early_stopping_patient:
                    logging.info(f'early stop at epoch{epoch}')
                    break

            # test
            self._test()

        self._load_checkpoint('best.pkl')
        self._test()

    def _train_epoch(self):
        self.model.train()
        for (image_w, _), label, index in tqdm(self.train_loader):
            image_w = image_w.to(self.device)
            label = label.to(self.device)
            index = index.to(self.device)
            clean_idx = self._clean_mask[index]

            self.optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logit = self.model(image_w)
                loss = F.cross_entropy(logit, label, reduction='none')
                loss = torch.where(clean_idx, loss, torch.zeros_like(loss)).mean()

                if self._ct_refine:
                    self.sample_selector.update(index, logit.softmax(-1).detach(), self.epoch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step()

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
            image = image.to(self.device)
            logit[index] = self.model(image).detach().cpu()
            labels[index] = label
        return logit, labels

    @torch.no_grad()
    def _fine(self):
        self.model.eval()

        # get feature
        features = []
        labels = []
        for img, label, idx in tqdm(self.eval_train_loader):
            img = img.to(self.device)
            labels.append(label)
            if self.args.arc == 'preact18':
                feature, _ = self.model(img, return_features=True)
                features.append(feature.cpu())
            else:
                raise ValueError(f'{self.args.arc} is not supported')

        labels = torch.cat(labels, dim=0).numpy()
        features = torch.cat(features, dim=0).numpy()

        # get singular vector
        singular_vector_dict = {}
        with tqdm(total=self.num_classes) as pbar:
            for index in range(self.num_classes):
                _, _, v = np.linalg.svd(features[labels == index])
                singular_vector_dict[index] = v[0]
                pbar.update(1)

        # get score
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat / np.linalg.norm(feat))) for indx, feat in
                  enumerate(tqdm(features))]
        scores = np.array(scores)

        # fit gmm in each class
        probs = np.zeros(len(labels))
        for c in range(self.num_classes):
            idx = labels == c
            s = scores[idx].reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)
            gmm.fit(s)
            p = gmm.predict_proba(s)
            probs[idx] = p[:, gmm.means_.argmax()]

        probs = torch.Tensor(probs)
        self._fine_clean_mask = probs.gt(self.p_threshold).to(self.device)

    def _sample_selection(self):
        # select clean samples by FINE
        if (self.epoch + 1) % 10 == 0:
            self._fine()
        self._clean_mask = self._fine_clean_mask
        # add samples selected by CT
        if self._ct_refine:
            self._ct_clean_mask = self.sample_selector.select_clean_labels(self.epoch)
            self._clean_mask = self._fine_clean_mask | self._ct_clean_mask
        logging.info(f'selected clean labels: {self._clean_mask.float().mean()}')
        if self.clean_or_not is not None:
            self._eval_sample_selection()

    def _eval_sample_selection(self):
        clean_pred = self._clean_mask.cpu().numpy()
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
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)

    def _load_checkpoint(self, name):
        checkpoint_path = os.path.join(self.args.save_dir, name)
        logging.info(f'loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])



