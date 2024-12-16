import torch
import torch.nn.functional as F
import numpy as np


def cal_margin(prediction, labels):
    labels = labels.unsqueeze(1)
    label_prediction = torch.gather(prediction, dim=1, index=labels).squeeze(1)
    # set the output to zero on labeled classes
    prediction2 = torch.scatter(prediction, dim=1, index=labels, value=0)
    max_prediction, max_idx = prediction2.max(dim=1)
    margin = label_prediction - max_prediction
    return margin


def metric(logit, target, topk=(1, 5)):
    output = logit.softmax(-1)
    _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
    correct = pred.eq(target.reshape(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(100 * correct_k / target.size(0))
    return res


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def knn(x, y, k=5):
    """
    :param x: Tensor(m, d)
    :param y: Tensor(n, d)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2.0).sum(dim=1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2.0).sum(dim=1, keepdim=True).expand(n, m).T
    dist = xx + yy - 2 * x @ y.T
    sorted_index = dist.argsort(dim=1, descending=False)
    return sorted_index[:, k]


def sharpen(x, temperature=0.5):
    x = x ** (1 / temperature)
    x = x / (x.sum(1, keepdim=True) + 1e-5)
    return x


def get_threshold(output, label, clean_mask):
    output = torch.scatter(output[clean_mask], dim=1, index=label[clean_mask].unsqueeze(1), value=0)
    res = output.max(dim=0)[0]
    return res
