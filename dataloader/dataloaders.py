from dataloader.cifar10_loader import CIFAR10Loader
from dataloader.cifar100_loader import CIFAR100Loader
from dataloader.clothing1m_loader import Clothing1mLoader
from dataloader.webvision_loader import WebVisionLoader
from dataloader.food101n_loader import Food101NLoader

def load_datasets(args):
    assert args.dataset in ['cifar-10', 'cifar-100', 'clothing1m', 'webvision', 'food101n'],\
        f'dataset {args.dataset} is not supported'
    if args.dataset == 'cifar-10':
        return CIFAR10Loader(args)
    elif args.dataset == 'cifar-100':
        return CIFAR100Loader(args)
    elif args.dataset == 'clothing1m':
        return Clothing1mLoader(args)
    elif args.dataset == 'food101n':
        return Food101NLoader(args)
    else:
        return WebVisionLoader(args)
