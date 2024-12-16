import numpy as np
from PIL import Image


def train_val_split(labels, validation_split, num_classes):
    labels = np.array(labels)
    train_n = int(len(labels) * (1 - validation_split) / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


def cifar_load_func(data):
    return Image.fromarray(data)


class MultiAug(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        res = [aug(x) for aug in self.transforms]
        return res
