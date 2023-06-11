import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging

# import torch.utils.data.sampler as sampler
import torch.utils.data as data
from tqdm import tqdm

from conf import settings
from utils import get_training_dataloader, get_test_dataloader

from sampling import *
from model import *
from logger import *

random_seed = 111
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
    
def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, indices in dataloader:
                yield img, label, indices
    else:
        while True:
            for img, _, _ in dataloader:
                yield img


def discrepancy(out1, out2):
    return torch.mean(torch.abs(out1.softmax(1) - out2.softmax(1)))


def train(epoch):

    net.train()
    FC.train()
    # unlabeled_loader = read_data(unlabeled_dataloader)
    for iter in tqdm(range(train_iterations)):
        # if iter<len(cifar10_labeled_loader):
        if True:
            images, labels, _ = next(labeled_loader)
            labels, images = labels.to(torch.device('cuda')), images.to(torch.device('cuda'))
            optimizer.zero_grad()
            outputs, mid = net(images)
            # out_1, out_2 = FC(mid)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_scheduler.step()
            # loss_1 = loss_function(out_1, labels)
            # loss_2 = loss_function(out_2, labels)
            # loss_c = loss_1 + loss_2
            # optim_fc.zero_grad()
            # loss_c.backward()
            # optim_fc.step()

        # if iter<len(unlabeled_dataloader):
        #     unlab_images, _, _ = next(unlabeled_loader)
        #     unlab_images = unlab_images.to(torch.device('cuda'))
        #     _, mid = net(images)
        #     out_1, out_2 = FC(mid)
        #     out_u, mid_u = net(unlab_images)
        #     out_1_u, out_2_u = FC(mid_u)

        #     loss_1 = loss_function(out_1, labels)
        #     loss_2 = loss_function(out_2, labels)
        #     loss_l = loss_1 + loss_2
        #     loss_u = discrepancy(out_1_u, out_2_u)
        #     loss_u1 = discrepancy(out_1_u, out_u)
        #     loss_u2 = discrepancy(out_2_u, out_u)
        #     loss_aux = loss_u + loss_u1 + loss_u2
        #     loss_comb = loss_l - loss_aux

        #     optim_fc.zero_grad()
        #     loss_comb.backward()
        #     optim_fc.step()

    logging.info(f'{epoch=} \tloss={loss.item()}')

@torch.no_grad()
def eval_training():

    net.eval()

    class_correct = [0.]*10; class_total = [0.]*10

    loop = tqdm(cifar10_test_loader, unit='batch', desc='| Test |', dynamic_ncols=True)
    for (x, y) in loop:

        x, y = x.to(torch.device('cuda')), y.to(torch.device('cuda'))

        outputs, _ = net(x)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == y)
        for i in range(len(c)):
            class_correct[y[i]] += c[i].item()
            class_total[y[i]] += 1

    logger.write_test_log(class_correct, class_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=20, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    args = parser.parse_args()
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, settings.TIME_NOW)
    os.makedirs(checkpoint_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(checkpoint_path, 'result.txt'))
    logger = Logger(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{rd}-{epoch}-{type}.pth')

    num_images = 50000
    initial_budget = 5000
    budget = 5000
    all_indices = set(np.arange(num_images))
    initial_indices = random.sample(all_indices, initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    current_indices = list(initial_indices)
    num_classes = 10
    cifar10_labeled_loader = get_training_dataloader(num_workers=4, batch_size=args.b, sampler=sampler)
    cifar10_test_loader = get_test_dataloader(num_workers=4, batch_size=args.b,)

    loss_function = nn.CrossEntropyLoss()
    # milestones = [60, 120, 160]
    for rd in range(10):
        logging.info(f'####################################round{rd}####################################')
        net = resnet18().to(torch.device('cuda'))
        FC = FullyConnected(num_classes).to(torch.device('cuda'))

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCH)
        # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
        # optim_fc = optim.SGD(FC.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # fc_train_scheduler = optim.lr_scheduler.MultiStepLR(optim_fc, milestones=milestones, gamma=0.2)
        # fc_train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_fc, T_max=settings.EPOCH)

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = get_training_dataloader(
            num_workers=os.cpu_count(),
            batch_size=args.b,
            sampler=unlabeled_sampler
            )
        iter_unlabeled = len(unlabeled_dataloader)
        iter_labeled = len(cifar10_labeled_loader)
        # if iter_unlabeled>iter_labeled:
        #     train_iterations = iter_unlabeled
        # else:
        #     train_iterations = iter_labeled
        train_iterations = iter_labeled
        labeled_loader = read_data(cifar10_labeled_loader)
        for epoch in range(1, settings.EPOCH+1):
            train(epoch)
        eval_training()
        logger.save_weight(net)
        if len(current_indices)>=50000:
            break
        mean_probs = labeled_samples(net, FC, cifar10_labeled_loader)
        sampled_indices = sample(net, FC, unlabeled_dataloader, budget, mean_probs)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        cifar10_labeled_loader = get_training_dataloader(
            num_workers=os.cpu_count(),
            batch_size=args.b,
            sampler=sampler
            )
    logger.show_result(0)