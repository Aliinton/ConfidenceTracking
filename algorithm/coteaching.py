import logging
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import metric
from torch.cuda.amp import GradScaler
from dataloader.dataloaders import load_datasets
from model.model import get_model
from sklearn.metrics import precision_score, recall_score, f1_score


class CoTeaching:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.n_epoch = args.n_epoch
        self.num_classes = args.num_classes
        self.warmup_epoch = args.warmup_epoch
        self.loader = load_datasets(args)
        self.train_loader = self.loader.run('train')
        self.valid_loader = self.loader.run('valid')
        self.test_loader = self.loader.run('test')
        self.drop_rate = 0.0
        self.noisy_label = torch.LongTensor(self.loader.train_noisy_labels)
        self.n_samples = len(self.noisy_label)
        args.n_samples = len(self.noisy_label)
        if args.dataset == 'cifar-10' and args.noise_type == 'asymmetric':
            self.noise_rate = args.noise_rate / 2
        else:
            self.noise_rate = args.noise_rate
        if self.loader.train_gt_labels is not None:
            args.clean_or_not = self.loader.train_gt_labels == self.loader.train_noisy_labels
            self.clean_or_not = args.clean_or_not
            logging.info(f'actual noise rate: {1 - self.clean_or_not.mean()}')
        else:
            self.clean_or_not = None
        self.model1 = get_model(args).to(args.device)
        self.model2 = get_model(args).to(args.device)
        self.optimizer = torch.optim.SGD(params=list(self.model1.parameters()) + list(self.model2.parameters()),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=args.milestones,
                                                              gamma=args.gamma)
        self.scaler = GradScaler()

        self._best_test_acc = 0.0
        self._test_acc = []
        self._best_valid_acc = 0.0
        self._early_stopping_cnt = 0

    def train(self):
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            logging.info(f'########## epoch{epoch + 1} ##########')
            logging.info(f'learning rate: {self.scheduler.get_last_lr()}')
            self.clean_mask1 = torch.zeros(len(self.noisy_label)).bool().to(self.device)
            self.clean_mask2 = torch.zeros(len(self.noisy_label)).bool().to(self.device)
            self._update_drop_rate(epoch)
            self._train_epoch()
            
            # sample selection evaluation
            logging.info('Net1 sample selection result:')
            self._eval_sample_selection(self.clean_mask1)
            logging.info('Net2 sample selection result:')
            self._eval_sample_selection(self.clean_mask2)

            # valid
            if self.valid_loader is not None:
                self._valid()
                if self._early_stopping_cnt > self.args.early_stopping_patient:
                    logging.info(f'early stop at epoch{epoch}')
                    break
            self._test()

        if self.valid_loader is not None:
            self._load_checkpoint('best.pkl')
            self._test()
            # only use one network
            logging.info(f'test network1')
            self._test2()

    def _eval_sample_selection(self, clean_pred):
        clean_pred = clean_pred.cpu().numpy()
        clean_true = self.clean_or_not
        p = precision_score(clean_true, clean_pred) * 100
        r = recall_score(clean_true, clean_pred) * 100
        f1 = f1_score(clean_true, clean_pred) * 100
        logging.info(f'selected clean labels: {clean_pred.sum()}')
        logging.info(f'sample selection result w.r.t clean labels: P: {p:.2f}, R: {r:.2f}, f1: {f1:.2f}')

    def _update_drop_rate(self, epoch):
        self.drop_rate = min(epoch/self.warmup_epoch * self.noise_rate, self.noise_rate)
        logging.info(f'drop rate: {self.drop_rate * 100}%')

    def _train_epoch(self):
        self.model1.train()
        self.model2.train()
        for (image_w, image_s), label, index in tqdm(self.train_loader):
            image_w = image_w.to(self.device)
            label = label.to(self.device)
            index = index.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logit1 = self.model1(image_w)
                logit2 = self.model2(image_w)
                loss = self._cal_loss(logit1, logit2, label, index)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step()

    def _cal_loss(self, logit1, logit2, label, index):
        batch_size = len(label)
        # get small-loss num
        n = int((1 - self.drop_rate) * batch_size)

        # get small-loss idx
        idx1 = F.cross_entropy(logit1, label, reduction='none').argsort()[:n]
        idx2 = F.cross_entropy(logit2, label, reduction='none').argsort()[:n]

        # cal loss
        loss1 = F.cross_entropy(logit1[idx2], label[idx2], reduction='none')
        loss2 = F.cross_entropy(logit2[idx1], label[idx1], reduction='none')
        loss = (loss1.sum() + loss2.sum()) / batch_size

        self.clean_mask1[index[idx1]] = True
        self.clean_mask2[index[idx2]] = True

        if self.args.noise_type == 'symmetric' and self.args.noise_rate == 0.8:
            prior = torch.ones(self.args.num_classes) / self.args.num_classes
            prior = prior.to(self.args.device)
            pred_mean1 = torch.softmax(logit1, dim=1).mean(0)
            penalty1 = torch.sum(prior * torch.log(prior / pred_mean1))
            pred_mean2 = torch.softmax(logit2, dim=1).mean(0)
            penalty2 = torch.sum(prior * torch.log(prior / pred_mean2))
            loss += (penalty1 + penalty2) / 2

        return loss

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
    def _test2(self):
        loader = self.test_loader
        self.model1.eval()
        logit = torch.zeros(len(loader.dataset), self.args.num_classes)
        labels = torch.zeros(len(loader.dataset)).long()
        for image, label, index in loader:
            image = image.to(self.device)
            output = self.model1(image)
            logit[index] = output.detach().cpu()
            labels[index] = label
        acc, _ = metric(logit, labels)
        self._test_acc.append(acc)
        self._best_test_acc = max(acc, self._best_test_acc)
        logging.info(f'test_acc: {acc}')
        logging.info(f'best_test_acc: {self._best_test_acc}')
        logging.info(f'last_test_acc: {sum(self._test_acc[-10:]) / len(self._test_acc[-10:])}')

    @torch.no_grad()
    def _infer(self, loader):
        self.model1.eval()
        self.model2.eval()
        logit = torch.zeros(len(loader.dataset), self.args.num_classes)
        labels = torch.zeros(len(loader.dataset)).long()
        for image, label, index in loader:
            image = image.to(self.device)
            output = (self.model1(image) + self.model2(image)) / 2
            logit[index] = output.detach().cpu()
            labels[index] = label
        return logit, labels

    def _save_checkpoint(self, name):
        checkpoint_path = os.path.join(self.args.save_dir, name)
        logging.info(f'saving checkpoint to {checkpoint_path}...')
        state = {
            'model1': self.model1.state_dict(),
            'model2': self.model2.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)

    def _load_checkpoint(self, name):
        checkpoint_path = os.path.join(self.args.save_dir, name)
        logging.info(f'loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path)
        self.model1.load_state_dict(checkpoint['model1'])
        self.model2.load_state_dict(checkpoint['model2'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
