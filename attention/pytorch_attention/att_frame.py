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
from att_model import TorchAttention, TextCNN, BiLSTM, ATTBiLSTM, MULTIBiLSTM
from dataloaders import att_dataloader
from sklearn.model_selection import train_test_split
import pandas as pd
Length = 400


def converts(temp):
    x_train = []
    for i in temp:
        lists = []
        lists[:0] = i
        lists = [int(i) for i in lists]
        x_train.append(lists)
    return x_train


class Att_Frame(nn.Module):
    def __init__(self, batch_size, lr, max_epoch):
        super().__init__()
        # self.model = TorchAttention().cuda()
        # self.model = TextCNN().cuda()
        # self.model = BiLSTM().cuda()
        # self.model = ATTBiLSTM().cuda()
        self.model = MULTIBiLSTM().cuda()
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        print("加载数据")
        # labels = np.loadtxt('label.txt')
        # encoded_seq = np.loadtxt('encoded_seq.txt')
        # encoded_seq_choose = encoded_seq[:, ((400 - Length) * 2):(1600 - (400 - Length) * 2)]
        # # print(encoded_seq_choose.shape)
        # x_train, x_test, y_train, y_test = train_test_split(encoded_seq_choose, labels, test_size=0.2)
        pd_train = pd.read_csv('num_train.csv')
        temp = pd_train['0']
        x_train = converts(temp)
        x_train = torch.tensor(x_train)
        y_train = pd_train['1']
        pd_test = pd.read_csv('num_test.csv')
        temp = pd_test['0']
        x_test = converts(temp)
        x_test = torch.tensor(x_test)
        y_test = pd_test['1']
        # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)  # 25000条样本
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
        print("加载完成")
        x_train = x_train.tolist()
        self.train_loader = att_dataloader(x_train, y_train, shuffle=True, batch_size=self.batch_size)
        self.test_loader = att_dataloader(x_test, y_test, shuffle=True, batch_size=self.batch_size)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_start(self):
        best_f1 = 0
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            # 把数据放进GPU
            for data in t:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                out = self.model(inputs)
                loss = self.loss_func(out, labels.float())
                train_loss += loss.item()
                t.set_postfix(loss=loss)
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
                        pre_set = np.round(pre)
                        # pre_set = np.where(pre == np.max(pre))[0][0]
                        # gold_set = np.where(gold == 1)[0][0]
                        gold_set = int(gold)
                        pred_num += 1
                        gold_num += 1
                        if pre_set == gold_set:
                            correct_num += 1
            print('正确个数', correct_num)
            print('预测个数', pred_num)
            precision = correct_num / pred_num
            recall = correct_num / gold_num
            f1_score = 2 * precision * recall / (precision + recall)
            if best_f1 < f1_score:
                best_f1 = f1_score
            # if precision > 0.9:
            #     # torch.save(self.model.state_dict(), 'D:/Projects/SCI/utils/1.pt')
            #     torch.save({'state_dict': self.model.state_dict()}, 'D:/Projects/SCI/utils/nyt.pth.tar')
            #     print("save successful")
            #     break
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
        print(best_f1)
