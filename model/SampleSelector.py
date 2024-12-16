import os
import torch
import numpy as np
from scipy.stats import norm
from multiprocessing import Pool
import logging
import math
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F

class CT:
    def __init__(self, args, labels):
        self.args = args
        self.device = args.device
        self.n_samples = len(labels)
        self.num_classes = args.num_classes
        self.n_epoch = args.n_epoch
        self.alpha = args.alpha
        self.labels = labels.to(self.device)

        self.margins = torch.zeros(self.n_samples, self.n_epoch, self.num_classes).to(self.device)
        self.score = torch.zeros(self.n_samples, self.num_classes).to(self.device)

    def update(self, index, pred, epoch):
        p = pred[torch.arange(len(index)), self.labels[index]]
        m = (p.view(-1, 1) - pred)

        # update margins
        self.margins[index, epoch] = m

        # update score
        temp = m.clone()
        self.score[index] += ((temp.unsqueeze(1) > self.margins[index, :epoch]).sum(1) - 
                              (temp.unsqueeze(1) < self.margins[index, :epoch]).sum(1))

    def select_clean_labels(self, epoch):
        n = epoch + 1
        self.score[torch.arange(self.n_samples), self.labels] = 1e6
        s, s_idx = self.score.min(1)

        # get var_s
        var_s = (n*(n-1)*(2*n+5)) / 18

        # get z
        z = torch.where(s.gt(0), (s - 1) / math.sqrt(var_s), (s + 1) / math.sqrt(var_s))
        z[s == 0] = 0

        # if all the margins are increasing, then this sample may be correctly labeled,
        res = z.gt(norm.ppf(1-self.alpha)).to(self.device)

        return res


class DIST(object):
    def __init__(self, args, labels):
        self.decay = args.decay
        self.device = args.device
        self.n_samples = args.n_samples
        self.labels = labels.to(self.device)
        self.threshold = torch.ones(args.n_samples).to(args.device) / args.n_samples
        self.pred = torch.ones(args.n_samples, args.num_classes).to(args.device)
        self.noise_mask = torch.zeros(args.n_samples).bool().to(args.device)

    def update(self, index, pred, epoch):
        self.pred[index] = pred
        self.threshold[index] = (self.decay * self.threshold[index] + (1 - self.decay) * pred.max(1)[0])

    def select_clean_labels(self, epoch):
        idx = torch.arange(len(self.labels))
        p = self.pred[idx, self.labels]
        res = p.gt(self.threshold)
        return res


class DIST_CT:
    def __init__(self, args, labels):
        self.args = args
        self.decay = args.decay
        self.device = args.device
        self.n_samples = args.n_samples
        self.n_epoch = args.n_epoch
        self.alpha = args.alpha
        self.labels = labels.to(self.device)
        self.clean_or_not = torch.Tensor(args.clean_or_not).bool().to(self.device)

        self.margins = torch.zeros(args.n_samples, args.n_epoch, args.num_classes).to(args.device)
        self.score = torch.zeros(args.n_samples, args.num_classes).to(args.device)

        self.threshold = torch.ones(args.n_samples).to(args.device) / args.n_samples
        self.pred = torch.ones(args.n_samples, args.num_classes).to(args.device) / args.num_classes

    def update(self, index, pred, epoch):
        # update DIST threshold
        self.pred[index] = pred
        self.threshold[index] = (self.decay * self.threshold[index] + (1 - self.decay) * pred.max(1)[0])

        # get current margin
        p = pred[torch.arange(len(index)), self.labels[index]]
        m = (p.view(-1, 1) - pred)

        # update margins
        self.margins[index, epoch] = m

        # update score
        self.score[index] += ((m.unsqueeze(1) > self.margins[index, :epoch]).sum(1) -
                              (m.unsqueeze(1) < self.margins[index, :epoch]).sum(1))

    def select_clean_labels(self, epoch):
        # select correct labels by DIST
        res = self.pred[torch.arange(self.n_samples), self.labels].gt(self.threshold)

        # add correct labels by CT
        self.score[torch.arange(self.n_samples), self.labels] = 1e6
        s, s_idx = self.score.min(1)

        # get var_s
        n = epoch + 1
        var_s = (n*(n-1)*(2*n+5)) / 18

        # get z
        z = torch.where(s.gt(0), (s - 1) / math.sqrt(var_s), (s + 1) / math.sqrt(var_s))
        z[s == 0] = 0

        # if all the margins are increasing, then this sample may be correctly labeled,
        res[z.gt(norm.ppf(1 - self.alpha))] = True
        return res


class GMM(object):
    def __init__(self, args, labels):
        self.args = args
        self.decay = args.decay
        self.device = args.device
        self.n_samples = args.n_samples
        self.labels = labels.to(self.device)
        self.pred = torch.ones(args.n_samples, args.num_classes).to(args.device)
        self.noise_mask = torch.zeros(args.n_samples).bool().to(args.device)
        self.reg_covar = args.reg_covar
        self.p_threshold = args.p_threshold

    def update(self, index, pred, epoch):
        self.pred[index] = pred

    def select_clean_labels(self, epoch):
        p = self.pred[torch.arange(len(self.labels)), self.labels]
        loss = -torch.log(p + 1e-6).detach().cpu()
        loss = (loss - loss.min()) / (loss.max() - loss.min())
        temp = loss.numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=self.reg_covar)
        gmm.fit(temp)
        prob = gmm.predict_proba(temp)
        prob = torch.Tensor(prob[:, gmm.means_.argmin()])
        res = torch.BoolTensor(prob > self.p_threshold).to(self.device)
        return res


class GMM_CT:
    def __init__(self, args, labels):
        self.args = args
        self.device = args.device
        self.n_samples = len(labels)
        self.num_classes = args.num_classes
        self.n_epoch = args.n_epoch
        self.alpha = args.alpha
        self.labels = labels.to(self.device)
        self.reg_covar = args.reg_covar
        self.p_threshold = args.p_threshold

        self.margins = torch.zeros(self.n_samples, self.n_epoch, self.num_classes).to(self.device)
        self.score = torch.zeros(self.n_samples, self.num_classes).to(self.device)
        self.pred = torch.ones(args.n_samples, args.num_classes).to(args.device)

    def update(self, index, pred, epoch):
        self.pred[index] = pred

        # get current margin
        p = pred[torch.arange(len(index)), self.labels[index]]
        m = (p.view(-1, 1) - pred)

        # update margins
        self.margins[index, epoch] = m

        # update score
        self.score[index] += ((m.unsqueeze(1) > self.margins[index, :epoch]).sum(1) -
                              (m.unsqueeze(1) < self.margins[index, :epoch]).sum(1))

    def select_clean_labels(self, epoch):
        # select correct labels by CT
        n = epoch + 1
        self.score[torch.arange(self.n_samples), self.labels] = 1e6
        s, s_idx = self.score.min(1)

        # get var_s
        var_s = (n * (n - 1) * (2 * n + 5)) / 18

        # get z
        z = torch.where(s.gt(0), (s - 1) / math.sqrt(var_s), (s + 1) / math.sqrt(var_s))
        z[s == 0] = 0

        # if all the margins are increasing, then this sample may be correctly labeled,
        res1 = z.gt(norm.ppf(1 - self.alpha))

        # select correct labels by GMM
        p = self.pred[torch.arange(len(self.labels)), self.labels]
        loss = -torch.log(p + 1e-6).detach().cpu()
        loss = (loss - loss.min()) / (loss.max() - loss.min())
        temp = loss.numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=self.reg_covar)
        gmm.fit(temp)
        prob = gmm.predict_proba(temp)
        prob = torch.Tensor(prob[:, gmm.means_.argmin()])
        res2 = torch.BoolTensor(prob > self.p_threshold).to(self.device)

        res = res1 | res2

        return res


class AUMSelector(object):
    def __init__(self, args, labels):
        self.args = args
        self.aum = torch.zeros(args.n_samples).to(args.device)
        self.labels = labels.to(args.device)

    def update_aum(self, batch_index, batch_logit):
        batch_label = self.labels[batch_index]
        self.aum[batch_index] += self.cal_margin(batch_logit, batch_label)

    def cal_margin(self, logit, label):
        label = label.unsqueeze(1)
        label_logit = torch.gather(logit, dim=1, index=label).squeeze(1)
        temp = torch.scatter(logit, dim=1, index=label, value=logit.min() - 1)
        max_logit, max_idx = temp.max(dim=1)
        margin = label_logit - max_logit
        return margin
    
    def select_clean_label(self, threshold_idx):
        threshold = torch.quantile(self.aum[threshold_idx], 0.99)
        res = self.aum.gt(threshold)
        res[threshold_idx] = False
        return res


class AUMSelector2(object):
    def __init__(self, args, labels):
        self.args = args
        self.aum = torch.zeros(args.n_samples).to(args.device)
        self.n_samples = args.n_samples
        self.labels = labels.to(args.device)
        self.delta = args.delta

        if args.dataset == 'cifar-10' and args.noise_type == 'asymmetric':
            self.noise_rate = args.noise_rate / 2
        else:
            self.noise_rate = args.noise_rate

    def update(self, batch_index, batch_logit, epoch):
        batch_label = self.labels[batch_index]
        self.aum[batch_index] += self.cal_margin(batch_logit, batch_label)

    def cal_margin(self, logit, label):
        label = label.unsqueeze(1)
        label_logit = torch.gather(logit, dim=1, index=label).squeeze(1)
        temp = torch.scatter(logit, dim=1, index=label, value=logit.min() - 1.0)
        max_logit, max_idx = temp.max(dim=1)
        margin = label_logit - max_logit
        return margin

    def select_clean_labels(self, epoch):
        sorted_idx = torch.argsort(self.aum, descending=True).cpu()
        res = torch.zeros(self.args.n_samples).bool().to(self.args.device)
        clean_num = int(self.n_samples * (1 - self.noise_rate - self.delta))
        res[sorted_idx[:clean_num]] = True
        return res


class AUMSelector2_CT(object):
    def __init__(self, args, labels):
        self.args = args
        self.num_classes = args.num_classes
        self.n_samples = args.n_samples
        self.device = args.device
        self.delta = args.delta
        self.aum = torch.zeros(args.n_samples).to(args.device)
        self.labels = labels.to(args.device)

        if args.dataset == 'cifar-10' and args.noise_type == 'asymmetric':
            self.noise_rate = args.noise_rate / 2
        else:
            self.noise_rate = args.noise_rate

        self.alpha = args.alpha
        self.margins = torch.zeros(args.n_samples, args.n_epoch, args.num_classes, dtype=torch.float16).to(args.device)
        self.score = torch.zeros(args.n_samples, args.num_classes).to(args.device)

    def update(self, index, batch_logit, epoch):
        # update aum
        batch_label = self.labels[index]
        self.aum[index] += self.cal_margin(batch_logit, batch_label)

        # update mk-score
        pred = batch_logit.softmax(1).detach()
        p = pred[torch.arange(len(index)), self.labels[index]]
        m = p.view(-1, 1) - pred
        self.margins[index, epoch] = m.to(torch.float16)
        delta = ((m.unsqueeze(1) > self.margins[index, :epoch]).sum(1) -
                 (m.unsqueeze(1) < self.margins[index, :epoch]).sum(1)).float()
        self.score[index] = self.score[index] + delta

    def cal_margin(self, logit, label):
        label = label.unsqueeze(1)
        label_logit = torch.gather(logit, dim=1, index=label).squeeze(1)
        temp = torch.scatter(logit, dim=1, index=label, value=logit.min() - 1)
        max_logit, max_idx = temp.max(dim=1)
        margin = label_logit - max_logit
        return margin

    def select_clean_labels(self, epoch):
        # sample selection by AUM
        sorted_idx = torch.argsort(self.aum, descending=True).cpu()
        aum_clean_mask = torch.zeros(self.args.n_samples).bool().to(self.args.device)
        clean_num = int(self.n_samples * (1 - self.noise_rate - self.delta))
        aum_clean_mask[sorted_idx[:clean_num]] = True
        # aum_clean_mask = self.aum.gt(0)

        # sample selection by CT
        idx = torch.arange(self.n_samples).to(self.device)
        self.score[idx, self.labels] = 1e8
        s, s_idx = self.score.min(1)
        n = (epoch + 1)
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        z = torch.where(s.gt(0), (s - 1) / math.sqrt(var_s), (s + 1) / math.sqrt(var_s))
        z[s == 0] = 0
        ct_clean_mask = z.gt(norm.ppf(1 - self.alpha))

        res = aum_clean_mask | ct_clean_mask
        return res


class L2DSelector(torch.nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, n_layer=2, n_class=2, dropout=0):
        super(L2DSelector, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.in_dim = in_dim
        self.classifier = torch.nn.Linear(hidden_dim, n_class)
        self.lstm = torch.nn.LSTM(self.in_dim, hidden_dim, n_layer, batch_first=True, dropout=self.dropout, bidirectional=False)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out
