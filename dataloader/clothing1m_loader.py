import os.path
import logging
import random

import torch
from PIL import Image
from sklearn.mixture import GaussianMixture
from torchvision import transforms
from dataloader.utils import MultiAug
from dataloader.lnl_dataset import LNLDataset
from torch.utils.data import DataLoader


class Clothing1mLoader:
    def __init__(self, args):
        self.args = args
        self.warmup_batch_size = args.batch_size * 2
        self.train_batch_size = args.batch_size
        self.eval_batch_size = args.batch_size
        self.data_dir = os.path.join(args.data_dir, 'clothing1M')
        # define transforms
        mean = (0.6959, 0.6537, 0.6371)
        std = (0.3113, 0.3192, 0.3214)
        self.transform_weak = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.transform_strong = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.transform_infer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # get labels
        self.noisy_labels = dict()
        self.clean_labels = dict()
        with open(os.path.join(self.data_dir, 'noisy_label_kv.txt'), 'r') as f:
            for l in f.readlines():
                entry = l.strip().split()
                img_path = os.path.join(self.data_dir, entry[0])
                self.noisy_labels[img_path] = int(entry[1])
        with open(os.path.join(self.data_dir, 'clean_label_kv.txt'), 'r') as f:
            for l in f.readlines():
                entry = l.strip().split()
                img_path = os.path.join(self.data_dir, entry[0])
                self.clean_labels[img_path] = int(entry[1])

        # train data
        self.train_images = []
        self.train_gt_labels = None
        self.train_noisy_labels = []
        with open(os.path.join(self.data_dir, 'noisy_train_key_list.txt'), 'r') as f:
            for idx, l in enumerate(f.readlines()):
                img_path = os.path.join(self.data_dir, l.strip())
                self.train_images.append(img_path)
                self.train_noisy_labels.append(self.noisy_labels[img_path])

        train_idx = self.sample_balanced_subset(100000)
        self.train_images = [self.train_images[i] for i in train_idx]
        self.train_noisy_labels = [self.train_noisy_labels[i] for i in train_idx]

        # valid data
        self.valid_images = []
        self.valid_labels = []
        with open(os.path.join(self.data_dir, 'clean_val_key_list.txt'), 'r') as f:
            for l in f.readlines():
                img_path = os.path.join(self.data_dir, l.strip())
                self.valid_images.append(img_path)
                self.valid_labels.append(self.clean_labels[img_path])

        # test data
        self.test_images = []
        self.test_labels = []
        with open(os.path.join(self.data_dir, 'clean_test_key_list.txt'), 'r') as f:
            for l in f.readlines():
                img_path = os.path.join(self.data_dir, l.strip())
                self.test_images.append(img_path)
                self.test_labels.append(self.clean_labels[img_path])

        logging.info(f'Train: {len(self.train_images)}, Val: {len(self.valid_images)}, Test: {len(self.test_images)}')

    def sample_balanced_subset(self, n_samples):
        res = []
        shuffle_idx = torch.randperm(len(self.train_images))
        class_num = torch.zeros(self.args.num_classes)
        for idx in shuffle_idx:
            label = self.train_noisy_labels[idx]
            if class_num[label] < (n_samples / self.args.num_classes) and len(res) < n_samples:
                res.append(idx)
                class_num[label] += 1
        return torch.LongTensor(res)

    def run(self, mode, clean_mask=None, train_idx=None, num_batches=None):
        if mode == 'train':
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=MultiAug([self.transform_weak, self.transform_strong]),
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.train_batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
            return loader
        elif mode == 'warmup':
            warmup_idx = self.sample_balanced_subset(num_batches * self.warmup_batch_size)
            logging.info(f'|warmup set|: {len(warmup_idx)}.')
            dataset = LNLDataset(images=[self.train_images[i] for i in warmup_idx],
                                 observed_label=[self.train_noisy_labels[i] for i in warmup_idx],
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_weak,
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.warmup_batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
            return loader

        elif mode == 'eval_train':
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_infer,
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.eval_batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            return loader

        elif mode == 'ssl':
            assert clean_mask is not None

            labeled_idx = []
            unlabeled_idx = []
            for i in train_idx:
                if clean_mask[i]:
                    labeled_idx.append(i)
                else:
                    unlabeled_idx.append(i)

            logging.info(f'|labeled set|: {len(labeled_idx)}, |unlabeled_set|: {len(unlabeled_idx)}')

            labeled_dataset = LNLDataset(images=[self.train_images[i] for i in labeled_idx],
                                         observed_label=[self.train_noisy_labels[i] for i in labeled_idx],
                                         load_func=lambda x: Image.open(x).convert('RGB'),
                                         transform=MultiAug([self.transform_weak, self.transform_weak]),
                                         ground_truth_labels=None)
            labeled_loader = DataLoader(dataset=labeled_dataset,
                                        batch_size=self.train_batch_size,
                                        shuffle=True,
                                        num_workers=8,
                                        pin_memory=True)


            unlabeled_dataset = LNLDataset(images=[self.train_images[i] for i in unlabeled_idx],
                                           observed_label=[self.train_noisy_labels[i] for i in unlabeled_idx],
                                           load_func=lambda x: Image.open(x).convert('RGB'),
                                           transform=MultiAug([self.transform_weak, self.transform_weak]),
                                           ground_truth_labels=None)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                          batch_size=self.train_batch_size,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)

            return labeled_loader, unlabeled_loader

        elif mode == 'valid':
            dataset = LNLDataset(images=self.valid_images,
                                 observed_label=self.valid_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_infer,
                                 ground_truth_labels=self.valid_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.eval_batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            return loader
        elif mode == 'test':
            dataset = LNLDataset(images=self.test_images,
                                 observed_label=self.test_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_infer,
                                 ground_truth_labels=self.test_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.eval_batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            return loader
