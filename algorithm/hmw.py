import logging
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.dataloaders import load_datasets
from model.model import get_model
from utils import metric
from torch.cuda.amp import GradScaler
import numpy as np



class NaturalDistanceWeighting(nn.Module):

    def __init__(self, num_classes, feat_dim, train_size, train_epoch, warmup_epoch, alpha=100., beta=100., top_rate=0.01, bias=False,
                 if_aum=1, if_anneal=1, if_spherical=1) -> None:
        super(NaturalDistanceWeighting, self).__init__()
        self.feat_dim = feat_dim
        self.top_rate = top_rate
        self.num_classes = num_classes
        self.train_epoch = train_epoch
        self.warmup_epoch = warmup_epoch
        self.add_weights = torch.zeros(train_epoch, train_size).cuda()
        #self.alpha = nn.Parameter(torch.randn(1, 1))
        #self.beta = nn.Parameter(torch.randn(1, 1))
        #self.a = nn.Parameter(torch.randn(1, 1))
        self.alpha = alpha
        self.beta = beta
        self.if_aum = if_aum
        self.if_anneal = if_anneal
        self.if_spherical = if_spherical

        #print('topk=', np.maximum(int((self.num_classes - 2) * self.top_rate), 1))
        self.weight = nn.Parameter(torch.empty((num_classes, feat_dim)))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        print('alpha=', self.alpha)
        print('beta=', self.beta)
        #print('s=', self.s)
        print('top_k=',np.maximum(int((self.num_classes - 2) * self.top_rate), 1))
        print('if_aum={}, if_anneal={}, if_spherical={}'.format(self.if_aum, self.if_anneal, self.if_spherical))

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, feat, labels=None, idx=None, ep=None, mixup=False, labels_a=None, labels_b=None, mix_rate=None):

        logits = F.linear(feat, self.weight, self.bias)
        s_logits = torch.softmax(logits, dim=-1)

        nfeat = F.normalize(feat, dim=1)
        nmeans = F.normalize(self.weight, dim=1)

        feat_dis = nfeat.unsqueeze(dim=1)
        means_dis = nmeans.unsqueeze(dim=0)

        if self.if_spherical == 1:
            angle_distances = (- torch.sum(feat_dis * means_dis, dim=-1) + 1.) / 2
        else:
            angle_distances = - F.softmax(logits, dim=-1)

        if labels is None:
            return logits


        else:
            if mixup == False:
                labeled_angle_distance = torch.gather(angle_distances, dim=-1, index=labels.unsqueeze(dim=-1))

                # min_angle_distance = torch.min(angle_distances, dim=-1, keepdim=True).values

                arrange_idxs = torch.arange(0, self.num_classes).unsqueeze(dim=0).tile((len(labels), 1)).cuda()

                other_idxs = torch.empty(len(labels), self.num_classes - 1, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_idxs[i] = arrange_idxs[i][arrange_idxs[i] != labels[i]]

                other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_idxs)

                min_other_angle = torch.min(other_angle_distances, dim=-1, keepdim=True)

                min_other_angle_distance = min_other_angle.values
                min_other_angle_idxs = min_other_angle.indices

                other_other_idxs = torch.empty(len(labels), self.num_classes - 2, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_other_idxs[i] = torch.cat(
                        (other_idxs[i][0: min_other_angle_idxs[i]], other_idxs[i][min_other_angle_idxs[i] + 1:]), dim=0)

                other_other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_other_idxs)

                top_k = np.maximum(int((self.num_classes - 2) * self.top_rate), 1)

                other_other_topk = torch.sort(other_other_angle_distances, dim=-1).values[:, :top_k]

                other_other_average_distances = other_other_topk.mean(dim=-1)

                assert torch.min(other_other_average_distances.squeeze() - min_other_angle_distance.squeeze()) >= 0

                if self.if_aum == 1:
                    weights = torch.sigmoid(self.alpha * (min_other_angle_distance.squeeze() - labeled_angle_distance.squeeze())) \
                              + torch.exp(self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))

                    weights = weights / 2.

                else:
                    weights = torch.exp(self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))


                self.add_weights[ep, idx] = weights.squeeze().detach()

                descends = torch.softmax(
                    torch.tensor([1 + np.cos((i / self.train_epoch) * np.pi) for i in range(ep + 1)]), dim=0).cuda().unsqueeze(dim=-1)

                logits = F.normalize(logits, dim=-1)

                #final_weights = (self.add_weights[:, idx] / (ep + 1)).squeeze().unsqueeze(dim=-1)
                if self.if_anneal == 1:
                    final_weights = (self.add_weights[:ep + 1, idx] * descends).sum(dim=0).squeeze().unsqueeze(dim=-1)
                else:
                    final_weights = weights.squeeze().unsqueeze(dim=-1)

            else:
                labeled_angle_distance_a = torch.gather(angle_distances, dim=-1, index=labels_a.unsqueeze(dim=-1))
                labeled_angle_distance_b = torch.gather(angle_distances, dim=-1, index=labels_b.unsqueeze(dim=-1))
                # labeled_angle_distance = mix_rate * labeled_angle_distance_a + (1 -mix_rate) * labeled_angle_distance_b
                labeled_angle_distance = torch.minimum(labeled_angle_distance_a, labeled_angle_distance_b)

                # min_angle_distance = torch.min(angle_distances, dim=-1, keepdim=True).values

                arrange_idxs = torch.arange(0, self.num_classes).unsqueeze(dim=0).tile((len(labels), 1)).cuda()

                other_idxs = torch.empty(len(labels), self.num_classes - 2, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_idx = arrange_idxs[i][arrange_idxs[i] != labels_a[i]]
                    other_idx = other_idx[other_idx != labels_b[i]]
                    if len(other_idx) == self.num_classes - 1:
                        pop_idx = np.random.randint(low=0, high=self.num_classes - 1)
                        other_idx = torch.cat((other_idx[0:pop_idx],other_idx[pop_idx+1:]), dim=0)
                    other_idxs[i] = other_idx

                other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_idxs)

                min_other_angle = torch.min(other_angle_distances, dim=-1, keepdim=True)

                min_other_angle_distance =  min_other_angle.values
                min_other_angle_idxs =  min_other_angle.indices

                other_other_idxs = torch.empty(len(labels), self.num_classes - 3, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_other_idxs[i] = torch.cat((other_idxs[i][0: min_other_angle_idxs[i]], other_idxs[i][min_other_angle_idxs[i] + 1:]), dim=0)

                other_other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_other_idxs)

                top_k = np.maximum(int((self.num_classes - 3) * self.top_rate), 1)

                other_other_topk = torch.sort(other_other_angle_distances, dim=-1).values[:,:top_k]

                other_other_average_distances = other_other_topk.mean(dim=-1)

                assert torch.min(other_other_average_distances.squeeze() - min_other_angle_distance.squeeze()) >= 0

                if self.if_aum == 1:

                    weights = torch.sigmoid(
                        self.alpha * (min_other_angle_distance.squeeze() - labeled_angle_distance.squeeze())) \
                              + torch.exp(
                        self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))

                    weights = weights / 2.

                else:
                    weights = torch.exp(self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))
                #logits = F.normalize(logits, dim=-1)

                revise_distance = torch.sort(other_angle_distances, dim=-1).values[:, 1]

                revise_idxs = torch.where(revise_distance < labeled_angle_distance.squeeze())[0]

                f_weights = torch.ones_like(weights)
                f_weights[revise_idxs] = weights[revise_idxs]

                final_weights = f_weights.squeeze().unsqueeze(dim=-1)

            out_weights = final_weights.detach()

            return logits, out_weights



class HMW:
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

        
        # sample weighter
        self.weighting = NaturalDistanceWeighting(num_classes=self.num_classes, 
                                                  feat_dim=512,
                                                  train_size=self.n_samples,
                                                  train_epoch=self.n_epoch,
                                                  warmup_epoch=self.warmup_epoch,
                                                  alpha=100, 
                                                  beta=100,
                                                  top_rate=0.01)
        self.weighting = self.weighting.to(args.device)

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
        self.weighting.train()
        for (image_w, _), label, index in tqdm(self.train_loader):
            image_w = image_w.to(self.device)
            label = label.to(self.device)
            index = index.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feat, _ = self.model(image_w, return_features=True)
                logit, weight = self.weighting(feat, label, idx=index, ep=self.epoch)
                if self.epoch < self.warmup_epoch:
                    loss = F.cross_entropy(logit, label)
                else:
                    loss = (F.cross_entropy(logit, label, reduction='none') * weight.squeeze()).mean()

            if self.args.noise_type == 'asymmetric':
                loss += self._conf_penalty(logit)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step()
    
    def _conf_penalty(self, x):
        probs = x.softmax(1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

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
        self.weighting.eval()
        logit = torch.zeros(len(loader.dataset), self.args.num_classes)
        labels = torch.zeros(len(loader.dataset)).long()
        for image, label, index in loader:
            image = image.to(self.device)
            feat, _ = self.model(image, return_features=True)
            logit[index] = self.weighting(feat).detach().cpu()
            labels[index] = label
        return logit, labels


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
