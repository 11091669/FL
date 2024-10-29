import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset
import random
from torch.utils.data import random_split

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


# 定义一个函数将单通道图像复制三次以创建三通道图像
def to_three_channels(x):
    return x.repeat(3, 1, 1)

def data_set(n, train_data, test_data):    
    size = [1/n]*n
    trainsets = random_split(train_data, size)
    return trainsets, test_data

def data_set_asy(n, train_data, test_data):
    train_labels = np.array(train_data.targets)
    client_idcs = dirichlet_split_noniid(train_labels, alpha=1, n_clients=n)
    trainsets= []
    for i in range(n):
        trainsets.append(Subset(train_data,client_idcs[i]))
    return trainsets, test_data

def get_cifar10(n, mode):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    if mode == 0:
        return data_set(n, train_data, test_data)
    else: return data_set_asy(n, train_data, test_data)

def get_mnist(n,mode):
    #数据集预处理参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(to_three_channels)  # 将单通道图像复制三次以创建三通道图像
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    if mode == 0:
        return data_set(n, train_data, test_data)
    else: return data_set_asy(n, train_data, test_data)