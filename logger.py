import os
import matplotlib.pyplot as plt
import csv
import logging

import torch
from torch.utils.data import DataLoader
from matplotlib import font_manager

class Logger():

    def __init__(self, log_path):
        self.log_path = log_path
        self.train_acc = []; self.train_loss = []
        self.test_acc = []
        self.rd = 0
        font_manager.fontManager.addfont("/home/ueno/fonts/times.ttf")
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['mathtext.fontset'] = 'stix'

    def save_weight(self, clf):
        if 'module' in [i for i,_ in clf.named_parameters()][0].split('.'):
            torch.save(clf.module.state_dict(), self.log_path+'/weight'+str(self.rd)+'.pth')
        else:
            torch.save(clf.state_dict(), self.log_path+'/weight'+str(self.rd)+'.pth')

    def write_train_log(self):
        for i in range(len(self.train_acc)):
            logging.info('Epoch:' + str(i) + '- training accuracy:'+str(self.train_acc[i])+'- training loss:'+str(self.train_loss[i]))
        self.train_acc = []; self.train_loss = []

    def write_test_log(self, class_correct, class_total):
        self.rd += 1
        self.test_acc.append(sum(class_correct)/sum(class_total))

        logging.info(f'test accuracy: {round(self.test_acc[-1]*100, 2)}')

    def show_result(self, seed):
        with open(self.log_path+'/../../test.csv', mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([seed])
            writer.writerow(self.test_acc)