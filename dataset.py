import os
import pickle

import numpy
from torch.utils.data import Dataset
from torchvision import datasets


class CIFAR10Train(Dataset):

    def __init__(self, path, transform=None):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=transform)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index


class CIFAR10Test(Dataset):

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar10:
            self.data = pickle.load(cifar10, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image