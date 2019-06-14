# -*- coding: utf-8 -*-
# This code was created with reference to followings
#   https://github.com/pytorch/examples/tree/master/imagenet
#   https://github.com/CuriousAI/mean-teacher/tree/master/pytorch
#   https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f
#   https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

import os
import torch
import torchvision.transforms as transforms
# import torchvision.datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np

NO_LABEL = -1

def datasetload_SSL(datasets, args):
    """args:
            data_dir, batch_ratio
    """
    if datasets == 'ImageNet':
        datapath = os.path.expanduser(args.data_dir)
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomRotation(10),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                              transforms.ToTensor(),
                                              normalize])
        val_transform  = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalize])
        label_train_dataset   = torchvision.datasets.ImageFolder(traindir, train_transform)
        label_indices, unlabel_indices = split_label_unlabel(label_train_dataset.targets,
                                                             1000,
                                                             args.num_labeled_per_classes)
        label_sampler  = SubsetRandomSampler(label_indices)
        unlabel_sampler =  SubsetRandomSampler(unlabel_indices)
        unlabel_train_dataset = torchvision.datasets.ImageFolder(traindir,
                                                                 MultiTransform(train_transform, args.K))
        val_dataset = torchvision.datasets.ImageFolder(valdir, val_transform)

    elif 'cifar' in datasets:
        datapath = os.path.expanduser(args.data_dir)
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
        val_transform = transforms.Compose([transforms.ToTensor(),
                                            normalize])
        if datasets == 'cifar10':
            label_train_dataset   = CIFAR10(root=datapath, train=True, download=True,
                                            transform=train_transform)
            unlabel_train_dataset = CIFAR10(root=datapath, train=True, download=True,
                                            transform=MultiTransform(train_transform, args.K))
            label_indices, unlabel_indices = split_label_unlabel(label_train_dataset.targets,
                                                                 10,
                                                                 args.num_labeled_per_classes)
            label_sampler  = SubsetRandomSampler(label_indices)
            unlabel_sampler =  SubsetRandomSampler(unlabel_indices)
            val_dataset   = CIFAR10(root=datapath, train=False, download=True,
                                    transform=val_transform)
        else: #cifar100
            label_train_dataset   = CIFAR100(root=datapath, train=True, download=True,
                                             transform=train_transform)
            unlabel_train_dataset = CIFAR100(root=datapath, train=True, download=True,
                                             transform=MultiTransform(train_transform, args.K))
            label_indices, unlabel_indices = split_label_unlabel(label_train_dataset.targets,
                                                                 100,
                                                                 args.num_labeled_per_classes)
            label_sampler  = SubsetRandomSampler(label_indices)
            unlabel_sampler =  SubsetRandomSampler(unlabel_indices)
            val_dataset   = CIFAR100(root=datapath, train=False, download=True,
                                     transform=val_transform)

    return label_train_dataset, unlabel_train_dataset, label_sampler, unlabel_sampler, val_dataset

## Function and Classes for SSL
class MultiTransform:
    def __init__(self, transform, K):
        self.transform = transform
        self.K = K

    def __call__(self, inp):
        out = [self.transform(inp) for _ in range(self.K)]
        return out


def split_label_unlabel(targets, num_classes, num_labeled_per_classes):
    targets = np.array(targets)
    label_indices, unlabel_indices = [], []

    for i in range(num_classes):
        idxs = np.where(targets == i)[0]
        np.random.shuffle(idxs)
        label_indices.extend(idxs[:num_labeled_per_classes])
        unlabel_indices.extend(idxs[num_labeled_per_classes:])
    return label_indices, unlabel_indices
