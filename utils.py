import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
from dataset import *


def get_training_dataloader(batch_size=16, num_workers=2, sampler=sampler):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar10_training = CIFAR10Train(path='/localdata/cifar10', transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, sampler=sampler, num_workers=num_workers, batch_size=batch_size, drop_last=False)

    return cifar10_training_loader

def get_test_dataloader(batch_size=16, num_workers=2, shuffle=False):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root='/localdata/cifar10', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader
