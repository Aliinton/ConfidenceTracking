import os.path

import torch
import random
import logging
from PIL import Image
from torchvision import transforms
from dataloader.utils import MultiAug
from dataloader.lnl_dataset import LNLDataset
from torch.utils.data import DataLoader
from dataloader.augment import TransformFixMatchLarge

class WebVisionLoader:
    def __init__(self, args):
        self.args = args
        self.warmup_batch_size = args.batch_size * 2
        self.train_batch_size = args.batch_size
        self.eval_batch_size = args.batch_size
        self.data_dir = os.path.join(args.data_dir, 'webvision')

        # define transform
        if 'DISC' in args.alg:
            mean = (0.6959, 0.6537, 0.6371)
            std = (0.3113, 0.3192, 0.3214)
            self.transform_train = TransformFixMatchLarge(mean, std)

            self.transform_infer = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.transform_infer_imagenet = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            self.transform_weak = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.transform_strong = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.transform_train = MultiAug([self.transform_weak, self.transform_weak])
            
            self.transform_infer = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.transform_infer_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        # train data
        self.train_images = []
        self.train_gt_labels = None
        self.train_noisy_labels = []
        with open(os.path.join(self.data_dir, 'info/train_filelist_google.txt'), 'r') as f:
            for l in f.readlines():
                img, label = l.strip().split(' ')
                label = int(label)
                if label < 50:
                    self.train_images.append(os.path.join(self.data_dir, img))
                    self.train_noisy_labels.append(label)

        # valida data
        self.valid_images = []
        self.valid_labels = []
        with open(os.path.join(self.data_dir, 'info/val_filelist.txt'), 'r') as f:
            for l in f.readlines():
                img, label = l.strip().split(' ')
                label = int(label)
                if label < 50:
                    self.valid_images.append(os.path.join(self.data_dir, 'val_images_256', img))
                    self.valid_labels.append(label)

        # test data
        test_data_dir = os.path.join(args.data_dir, 'ILSVRC2012')
        self.test_images = []
        self.test_labels = []
        with open(os.path.join(test_data_dir, 'imagenet_val.txt'), 'r') as f:
            for l in f.readlines():
                img, label = l.strip().split(' ')
                label = int(label)
                if label < 50:
                    self.test_images.append(os.path.join(test_data_dir, 'val', img))
                    self.test_labels.append(label)

        logging.info(f'Train: {len(self.train_images)}, Val: {len(self.valid_images)}, Test: {len(self.test_images)}')

    def run(self, mode, clean_mask=None, num_batches=None, train_idx=None):
        if mode == 'train':
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.open(x).convert('RGB'),
                                 transform=self.transform_train,
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.train_batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
            return loader
        elif mode == 'warmup':
            # sample warmup samples
            candidate_idx = list(range(len(self.train_images)))
            if num_batches is not None:
                warmup_idx = random.sample(candidate_idx, num_batches * self.warmup_batch_size)
            else:
                warmup_idx = candidate_idx

            # generate dataloader
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

            # generate labeled dataloader
            candidate_idx = torch.nonzero(clean_mask, as_tuple=False).squeeze(1).tolist()
            if num_batches is not None:
                labeled_idx = random.sample(candidate_idx, num_batches * self.train_batch_size)
            else:
                labeled_idx = candidate_idx
            labeled_dataset = LNLDataset(images=[self.train_images[i] for i in labeled_idx],
                                         observed_label=[self.train_noisy_labels[i] for i in labeled_idx],
                                         load_func=lambda x: Image.open(x).convert('RGB'),
                                         transform=self.transform_train,
                                         ground_truth_labels=None)
            labeled_loader = DataLoader(dataset=labeled_dataset,
                                        batch_size=self.train_batch_size,
                                        shuffle=True,
                                        num_workers=8,
                                        pin_memory=True)

            # generate unlabeled dataloader
            candidate_idx = torch.nonzero(~clean_mask, as_tuple=False).squeeze(1).tolist()
            if num_batches is not None:
                unlabeled_idx = random.sample(candidate_idx, num_batches * self.train_batch_size)
            else:
                unlabeled_idx = candidate_idx
            unlabeled_dataset = LNLDataset(images=[self.train_images[i] for i in unlabeled_idx],
                                           observed_label=[self.train_noisy_labels[i] for i in unlabeled_idx],
                                           load_func=lambda x: Image.open(x).convert('RGB'),
                                           transform=self.transform_train,
                                           ground_truth_labels=None)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                          batch_size=self.train_batch_size,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)
            
            logging.info(f'|labeled set|: {len(labeled_idx)}, |unlabeled_set|: {len(unlabeled_idx)}')

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
                                 transform=self.transform_infer_imagenet,
                                 ground_truth_labels=self.test_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.eval_batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            return loader
