import torch
from torch.utils.data import Dataset


class LNLDataset(Dataset):
    def __init__(self, images, observed_label, load_func, transform, ground_truth_labels=None, load_gt=False):
        super(LNLDataset).__init__()
        assert len(images) == len(observed_label)
        if ground_truth_labels is not None:
            assert len(images) == len(ground_truth_labels)
        self.load_gt = load_gt
        self.images = images
        self.observed_label = observed_label
        self.gt_labels = ground_truth_labels
        self.load_func = load_func
        self.transform = transform
        self._len = len(observed_label)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        img = self.images[index]
        label = self.observed_label[index]
        if self.load_func is not None:
            img = self.load_func(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.load_gt:
            gt_label = self.gt_labels[index]
            return img, label, gt_label, index
        else:
            return img, label, index
