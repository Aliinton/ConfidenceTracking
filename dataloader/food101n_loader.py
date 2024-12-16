import logging
import os.path
import torch
import numpy as np
from PIL import Image
from numpy.testing import assert_array_almost_equal
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader.utils import train_val_split, MultiAug
from dataloader.lnl_dataset import LNLDataset
from dataloader.augment import TransformFixMatchLarge


def load_func(x):
    return Image.open(x).convert('RGB')


class Food101NLoader:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.data_dir = args.data_dir
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size

        # define transform
        mean = (0.6959, 0.6537, 0.6371)
        std = (0.3113, 0.3192, 0.3214)
        self.transform_weak = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_strong = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_train = TransformFixMatchLarge(mean, std)
        self.transform_infer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # load classes
        self.class2id = dict()
        with open(os.path.join(args.data_dir, 'food101n', 'meta', 'classes.txt'), 'r') as f:
            for line in f:
                if line.strip() == 'class_name':
                    continue
                self.class2id[line.strip()] = len(self.class2id)

        # load train image
        self.train_images = []
        self.train_gt_labels = None
        self.train_noisy_labels = []
        for c, cid in self.class2id.items():
            class_dir = os.path.join(args.data_dir, 'food101n', 'images', c)
            for image in os.listdir(class_dir):
                self.train_images.append(os.path.join(class_dir, image))
                self.train_noisy_labels.append(cid)

        # load test images
        self.test_images = []
        self.test_labels = []
        with open(os.path.join(args.data_dir, 'food101', 'meta', 'test.txt'), 'r') as f:
            for line in f:
                image = line.strip()
                self.test_images.append(os.path.join(args.data_dir, 'food101', 'images', f'{image}.jpg'))
                self.test_labels.append(self.class2id[image.split('/')[0]])

        logging.info(f'Train: {len(self.train_images)}, Test: {len(self.test_images)}')

    def run(self, mode, clean_mask=None, num_batches=None, train_idx=None):
        if mode == 'train':
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_train,
                                 ground_truth_labels=self.train_gt_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True)
            return loader

        elif mode == 'test':
            dataset = LNLDataset(images=self.test_images,
                                 observed_label=self.test_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_infer,
                                 ground_truth_labels=self.test_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)
            return loader

        elif mode == 'warmup':
            # generate dataloader
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_weak,
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size * 2,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True)
            return loader

        elif mode == 'eval_train':
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_infer,
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)
            return loader

        elif mode == 'ssl':
            assert clean_mask is not None

            # generate labeled dataloader
            labeled_idx = torch.nonzero(clean_mask, as_tuple=False).squeeze(1).tolist()
            labeled_dataset = LNLDataset(images=[self.train_images[i] for i in labeled_idx],
                                         observed_label=[self.train_noisy_labels[i] for i in labeled_idx],
                                         load_func=lambda x: Image.open(x).convert('RGB'),
                                         transform=MultiAug([self.transform_weak, self.transform_weak]),
                                         ground_truth_labels=None)
            labeled_loader = DataLoader(dataset=labeled_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=16,
                                        pin_memory=True)

            # generate unlabeled dataloader
            unlabeled_idx = torch.nonzero(~clean_mask, as_tuple=False).squeeze(1).tolist()
            unlabeled_dataset = LNLDataset(images=[self.train_images[i] for i in unlabeled_idx],
                                           observed_label=[self.train_noisy_labels[i] for i in unlabeled_idx],
                                           load_func=lambda x: Image.open(x).convert('RGB'),
                                           transform=MultiAug([self.transform_weak, self.transform_weak]),
                                           ground_truth_labels=None)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=16,
                                          pin_memory=True)

            logging.info(f'|labeled set|: {len(labeled_idx)}, |unlabeled_set|: {len(unlabeled_idx)}')

            return labeled_loader, unlabeled_loader
