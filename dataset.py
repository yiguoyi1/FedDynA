import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, Dataset, DataLoader
import numpy as np
import os
import random
def get_dataset_loaders(dataset_name):
    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        # 不使用数据增强（无裁剪，无翻转）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # 获取数据集
        train_dataset = datasets.CIFAR10(root='cifar10data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='cifar10data', train=False, download=True,transform=transform)
    elif dataset_name == 'cifar100':
        # CIFAR-100 常用均值与标准差
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.CIFAR100(root='cifar100data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='cifar100data', train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        transform = transforms.ToTensor()
        train_dataset = datasets.FashionMNIST(root='dataset', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='dataset', train=False,download=True, transform=transform)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported!")

    train_size = int( len(train_dataset))
    test_size = int(len(test_dataset))

    train_indices = torch.randperm(len(train_dataset))[:train_size]
    test_indices = torch.randperm(len(test_dataset))[:test_size]

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset
