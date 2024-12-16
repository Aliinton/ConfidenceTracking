import logging
import os.path
import torch
import numpy as np
from PIL import Image
from scipy import stats
from math import inf
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader.utils import train_val_split, MultiAug
from dataloader.lnl_dataset import LNLDataset


class CIFAR10Loader:
    def __init__(self, args, download=True):
        self.args = args
        self.seed = args.seed
        self.data_dir = args.data_dir
        self.noise_type = args.noise_type
        self.noise_rate = args.noise_rate
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.valid = args.validation_split is not None

        # define transform
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        self.transforms_weak = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transforms_strong = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_train = MultiAug([self.transforms_weak, self.transforms_strong])
        self.transform_infer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # split train & valid dataset
        base_train = datasets.CIFAR10(args.data_dir, train=True, download=download)
        if self.valid:
            train_idx, val_idx = train_val_split(base_train.targets, args.validation_split, args.num_classes)
        else:
            train_idx = np.arange(len(base_train.targets))
            val_idx = None

        # train
        self.train_images = base_train.data[train_idx]
        self.train_gt_labels = np.array(base_train.targets)[train_idx]
        self.train_noisy_labels = self.corrupt_label(self.train_images, self.train_gt_labels, train_idx)

        # valid
        if self.valid:
            self.valid_images = base_train.data[val_idx]
            self.valid_gt_labels = np.array(base_train.targets)[val_idx]
            self.valid_noisy_labels = self.corrupt_label(self.valid_images, self.valid_gt_labels, val_idx)

        # test
        base_test = datasets.CIFAR10(args.data_dir, train=False, download=download)
        self.test_images = base_test.data
        self.test_labels = base_test.targets

        logging.info(f'Train: {len(self.train_images)}, Val: {len(self.valid_images) if self.valid else 0}, Test: {len(self.test_images)}')

    def corrupt_label(self, imgs, label, train_idx):
        assert self.noise_type in ['symmetric', 'asymmetric', 'clean', 'aggre', 'worst', 'rand1', 'rand2', 'rand3', 'instance'], \
            f'noise type {self.noise_type} is not supported'
        if self.noise_type == 'symmetric':
            noisy_label = self.symmetric_noise(label)
        elif self.noise_type == 'asymmetric':
            noisy_label = self.asymmetric_noise(label)
        elif self.noise_type == 'instance':
            noisy_label = self.instance_dependent_noise(imgs, label)
        else:
            noisy_label = self.realworld_noise(train_idx)
        return noisy_label
    
    def instance_dependent_noise(self, imgs, label, norm_std=0.1):
        flip_distribution = stats.truncnorm((0 - self.noise_rate) / norm_std, (1 - self.noise_rate) / norm_std, loc=self.noise_rate, scale=norm_std)
        q = flip_distribution.rvs(len(label))

        w = torch.FloatTensor(np.random.randn(self.num_classes, 3*32*32, self.num_classes))

        noisy_labels = []
        for i, (x, y) in enumerate(zip(imgs, label)):
            x = x.flatten()
            p_all = np.matmul(x, w[y])
            p_all[y] = -inf
            p_all = q[i] * F.softmax(torch.tensor(p_all),dim=0).numpy()
            p_all[y] = 1 - q[i]
            noisy_labels.append(np.random.choice(np.arange(self.num_classes), p=p_all/sum(p_all)))
        noisy_labels = np.array(noisy_labels)
        return noisy_labels

    def symmetric_noise(self, label):
        # assert 0. < self.noise_rate < 1.
        # res = label.copy()
        # indices = np.random.permutation(len(label))
        # for i, idx in enumerate(indices):
        #     if i < self.noise_rate * len(label):
        #         res[idx] = np.random.randint(self.num_classes, dtype=np.int32)
        # return res

        assert 0. < self.noise_rate < 1.
        P = np.ones((self.num_classes, self.num_classes)) * (self.noise_rate / (self.num_classes - 1))
        for i in range(self.num_classes):
            P[i, i] = 1 - self.noise_rate

        res = self.multiclass_noisify(label, P=P, random_state=self.seed)
        return res

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def asymmetric_noise(self, label):
        assert 0. < self.noise_rate < 1.
        res = label.copy()
        for i in range(10):
            indices = np.where(label == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.noise_rate * len(indices):
                    # truck -> automobile
                    if i == 9:
                        res[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        res[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        res[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        res[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        res[idx] = 7
        return res

    def realworld_noise(self, train_idx):
        noise_file = torch.load(os.path.join(self.args.data_dir, 'CIFAR-N', 'CIFAR-10_human.pt'))
        if self.args.noise_type == 'clean':
            return noise_file['clean_label'][train_idx]
        elif self.args.noise_type == 'aggre':
            return noise_file['aggre_label'][train_idx]
        elif self.args.noise_type == 'worst':
            return noise_file['worse_label'][train_idx]
        elif self.args.noise_type == 'rand1':
            return noise_file['random_label1'][train_idx]
        elif self.args.noise_type == 'rand2':
            return noise_file['random_label2'][train_idx]
        elif self.args.noise_type == 'rand3':
            return noise_file['random_label3'][train_idx]

    def run(self, mode, clean_mask=None, num_batches=None):
        if mode == 'train':
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.fromarray(x),
                                 transform=self.transform_train,
                                 ground_truth_labels=self.train_gt_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
            return loader

        elif mode == 'valid':
            if self.args.validation_split is None:
                return None
            dataset = LNLDataset(images=self.valid_images,
                                 observed_label=self.valid_noisy_labels,
                                 load_func=lambda x: Image.fromarray(x),
                                 transform=self.transform_infer,
                                 ground_truth_labels=self.valid_gt_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            return loader

        elif mode == 'test':
            dataset = LNLDataset(images=self.test_images,
                                 observed_label=self.test_labels,
                                 load_func=lambda x: Image.fromarray(x),
                                 transform=self.transform_infer,
                                 ground_truth_labels=self.test_labels)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            return loader
        
        elif mode == 'warmup':
            # generate dataloader
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.fromarray(x),
                                 transform=self.transforms_weak,
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size * 2,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
            return loader

        elif mode == 'eval_train':
            dataset = LNLDataset(images=self.train_images,
                                 observed_label=self.train_noisy_labels,
                                 load_func=lambda x: Image.fromarray(x),
                                 transform=self.transform_infer,
                                 ground_truth_labels=None)
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            return loader

        elif mode == 'ssl':
            assert clean_mask is not None

            # generate labeled dataloader
            labeled_idx = torch.nonzero(clean_mask, as_tuple=False).squeeze(1).tolist()
            labeled_dataset = LNLDataset(images=[self.train_images[i] for i in labeled_idx],
                                         observed_label=[self.train_noisy_labels[i] for i in labeled_idx],
                                         load_func=lambda x: Image.fromarray(x),
                                         transform=MultiAug([self.transforms_weak, self.transforms_weak]),
                                         ground_truth_labels=None)
            labeled_loader = DataLoader(dataset=labeled_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=8,
                                        pin_memory=True)

            # generate unlabeled dataloader
            unlabeled_idx = torch.nonzero(~clean_mask, as_tuple=False).squeeze(1).tolist()
            unlabeled_dataset = LNLDataset(images=[self.train_images[i] for i in unlabeled_idx],
                                           observed_label=[self.train_noisy_labels[i] for i in unlabeled_idx],
                                           load_func=lambda x: Image.fromarray(x),
                                           transform=MultiAug([self.transforms_weak, self.transforms_weak]),
                                           ground_truth_labels=None)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)

            return labeled_loader, unlabeled_loader


