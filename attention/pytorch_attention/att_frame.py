# _*_ coding: utf-8 _*_
# @Time : 2021/6/6 15:03
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_frame.py
# @desc :
import logging

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from keras.datasets import imdb
from att_model import TorchAttention
from dataloaders import att_dataloader
import pandas as pd


class Att_Frame(nn.Module):
    def __init__(self, batch_size, lr, max_epoch):
        super().__init__()
        self.model = TorchAttention().cuda()
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        print("加载数据")
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)  # 25000条样本
        print("加载完成")
        self.train_loader = att_dataloader(x_train, y_train, shuffle=True, batch_size=self.batch_size)
        self.test_loader = att_dataloader(x_test, y_test, shuffle=True, batch_size=self.batch_size)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_start(self):
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            # 把数据放进GPU
            for t_iter, data in enumerate(t):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                out = self.model(inputs)
                loss = self.loss_func(out, labels.float())
                train_loss += loss.item()
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = train_loss / len(t)
            logging.info(f"Epoch: {epoch}, train loss: {avg_loss}")
            self.model.eval()
            t = tqdm(self.test_loader)
            pred_num, gold_num, correct_num = 1e-10, 1e-10, 1e-10
            # dev_losses = 0
            with torch.no_grad():
                for iter_s, batch_samples in enumerate(t):
                    inputs, labels = batch_samples
                    inputs = inputs.cuda()
                    rel_out = self.model(inputs)
                    # 计算评价指标
                    labels = labels.numpy()
                    rel_out = rel_out.to('cpu').numpy()
                    for pre, gold in zip(rel_out, labels):
                        pre_set = np.where(pre == np.max(pre))[0][0]
                        gold_set = np.where(gold == 1)[0][0]
                        pred_num += 1
                        gold_num += 1
                        if pre_set == gold_set:
                            correct_num += 1
            print('正确个数', correct_num)
            print('预测个数', pred_num)
            precision = correct_num / pred_num
            recall = correct_num / gold_num
            f1_score = 2 * precision * recall / (precision + recall)
            # if precision > 0.9:
            #     # torch.save(self.model.state_dict(), 'D:/Projects/SCI/utils/1.pt')
            #     torch.save({'state_dict': self.model.state_dict()}, 'D:/Projects/SCI/utils/nyt.pth.tar')
            #     print("save successful")
            #     break
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))