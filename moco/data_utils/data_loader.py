import os

import torch
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from .tinyimagenet import TinyImageNet
from .augmentation import *

# Data loader
DATASETS = {'cifar10': CIFAR10, 'cifar100': CIFAR100, 'tiny-imagenet': TinyImageNet, 'imagenet': None}
MEAN = {'cifar10': [0.4914, 0.4822, 0.4465], 'cifar100': [0.5071, 0.4867, 0.4408], 'tiny-imagenet': [0.485, 0.456, 0.406], 'imagenet': [0.485, 0.456, 0.406]}
STD = {'cifar10': [0.2023, 0.1994, 0.2010], 'cifar100':[0.2675, 0.2565, 0.2761], 'tiny-imagenet': [0.229, 0.224, 0.225], 'imagenet': [0.229, 0.224, 0.225]}

__all__ = ['data_loader']


def data_loader(dataset, data_path, batch_size, num_workers, download=False, distributed=True, aug_plus=True, supervised=False):
    normalize = transforms.Normalize(MEAN[dataset], STD[dataset])

    # for self-supervised learning (pretrain)
    if not supervised:
        if aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]

        augmentation.insert(0, transforms.RandomResizedCrop(224, scale=(0.2, 1.)))

        train_transform = TwoCropsTransform(transforms.Compose(augmentation))

        if dataset == 'imagenet':
            traindir = os.path.join(data_path, 'train')
            train_dataset = ImageFolder(traindir, transform=train_transform)
        else:
            train_dataset = DATASETS[dataset](data_path, train=True, download=download, transform=train_transform)

        # for distributed learning
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        return train_loader, train_sampler
    
    # for supervised learning (lincls)
    else:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
        
        test_transform=transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])
        
        if dataset == 'imagenet':
            traindir = os.path.join(data_path, 'train')
            testdir = os.path.join(data_path, 'val')
            train_dataset = ImageFolder(traindir, transform=train_transform)
            test_dataset = ImageFolder(testdir, transform=test_transform)
        else:
            train_dataset = DATASETS[dataset](data_path, train=True, download=download, transform=train_transform)
            test_dataset = DATASETS[dataset](data_path, train=False, download=download, transform=test_transform)

        # for distributed learning
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader, train_sampler
