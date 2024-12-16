import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet_Zoo import ResNet, BasicBlock
from .PreResNet import PreActResNet, PreActBlock
from .InceptionResNetV2 import InceptionResNetV2
from torchvision.models import vgg19_bn


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)


def resnet50(num_classes=14):
    import torchvision.models as models
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def get_model(args):
    assert args.arc in ['resnet34', 'preact18', 'resnet50', 'InceptionResNetV2'], \
        f'model architecture {args.arc} is not supported'
    if args.arc == 'resnet34':
        return resnet34(args.num_classes)
    elif args.arc == 'preact18':
        return PreActResNet18(args.num_classes)
    elif args.arc == 'resnet50':
        return resnet50(args.num_classes)
    elif args.arc == 'InceptionResNetV2':
        return InceptionResNetV2(args.num_classes)
