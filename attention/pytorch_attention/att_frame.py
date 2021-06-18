# _*_ coding: utf-8 _*_
# @Time : 2021/6/6 15:03
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_frame.py
# @desc :
import codecs
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from att_model import MULTIBiLSTM, BiLSTM, TextCNN, TorchAttention
from dataloaders import att_dataloader

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
        pd_train = pd.read_csv('./single/num_train.csv')
        temp = pd_train['0'].values.tolist()
        # temp = [eval(i) for i in temp]
        train_temp = []
        for i in temp:
            train_temp.append([eval(j) for j in i])
        x_train = torch.tensor(train_temp)
        y_train = pd_train['1'].to_numpy()
        pd_test = pd.read_csv('./single/num_test.csv')
        temp_s = pd_test['0'].values.tolist()
        test_temp = []
        for i in temp_s:
            test_temp.append([eval(j) for j in i])
        # temp_s = [eval(i) for i in temp_s]
        x_test = torch.tensor(test_temp)
        y_test = pd_test['1'].to_numpy()
        # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)  # 25000条样本
        print("加载完成")
        self.train_loader = att_dataloader(x_train, y_train, shuffle=True, batch_size=self.batch_size)
        self.test_loader = att_dataloader(x_test, y_test, shuffle=True, batch_size=self.batch_size)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_start(self):
        best_f1, best_acc, best_recall = 0, 0, 0
        loss_log, acc_log, recall_log = [], [], []
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            # 把数据放进GPU
            is_test = False
            for data in t:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                out = self.model(inputs, is_test)
                # out = self.model(inputs)
                loss = self.loss_func(out, labels.float())
                train_loss += loss.item()
                t.set_postfix(loss=loss)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = train_loss / len(t)
            loss_log.append(avg_loss)
            logging.info(f"Epoch: {epoch}, train loss: {avg_loss}")
            self.model.eval()
            t = tqdm(self.test_loader)
            pred_num, gold_num, correct_num = 1e-10, 1e-10, 1e-10
            # dev_losses = 0
            with torch.no_grad():
                all_attention = []
                for iter_s, batch_samples in enumerate(t):
                    inputs, labels = batch_samples
                    inputs = inputs.cuda()
                    is_test = True
                    rel_out, attention = self.model(inputs, is_test)
                    # rel_out = self.model(inputs)
                    # 计算评价指标
                    labels = labels.numpy()
                    rel_out = rel_out.to('cpu').numpy()
                    # ——————————————————————————————————
                    idx = inputs.cpu().numpy()
                    piece_attention = plot_attention(attention)
                    for pre, gold, ids, att in zip(rel_out, labels, idx, piece_attention):
                        pre_set = np.round(pre)
                        gold_set = int(gold)
                        pred_num += 1
                        gold_num += 1
                        if pre_set == gold_set:
                            correct_num += 1
                            attention_temp_dict = {'inputs': ids.tolist(), 'labels': gold, 'attention': att.tolist()}
                            all_attention.append(attention_temp_dict)
                    # for pre, gold in zip(rel_out, labels):
                    #     pre_set = np.round(pre)
                    #     gold_set = int(gold)
                    #     pred_num += 1
                    #     gold_num += 1
                    #     if pre_set == gold_set:
                    #         correct_num += 1
                    # inputs 和 attention
            print('正确个数', correct_num)
            print('预测个数', pred_num)
            precision = correct_num / pred_num
            recall = correct_num / gold_num
            f1_score = 2 * precision * recall / (precision + recall)
            acc_log.append(precision)
            recall_log.append(recall)
            if best_f1 < f1_score:
                best_f1, best_acc, best_recall = f1_score, precision, recall
                with codecs.open('attention_sample.json', 'w', encoding='utf-8') as f:
                    json.dump(all_attention, f, indent=4, ensure_ascii=False)
            # if precision > 0.9:
            #     # torch.save(self.model.state_dict(), 'D:/Projects/SCI/utils/1.pt')
            #     torch.save({'state_dict': self.model.state_dict()}, 'D:/Projects/SCI/utils/nyt.pth.tar')
            #     print("save successful")
            #     break
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
        print(best_f1, best_acc, best_recall)
        with codecs.open('bi_mul_train_loss.json', 'w', encoding='utf-8') as f:
            json.dump(loss_log, f, indent=4, ensure_ascii=False)
        with codecs.open('bi_mul_train_acc.json', 'w', encoding='utf-8') as f:
            json.dump(acc_log, f, indent=4, ensure_ascii=False)
        with codecs.open('bi_mul_train_recall.json', 'w', encoding='utf-8') as f:
            json.dump(recall_log, f, indent=4, ensure_ascii=False)


def plot_attention(attention):
    # attention ==> batch_size, num layer,  sequence len,  sequence len
    # 69,70 是剪切位点sas
    # 138 个输入，每个输入 的 对应全句的attention系数
    # 只需要剪切位点的两个 batch_size, num_layer,  2, 138
    attention = attention[:, :, 68:70, :]
    # 合并多头, batch_size , 2, 138
    attention = torch.mean(attention, 1)
    # 合并剪切位点，mean一下，batch，138
    attention = torch.mean(attention, 1)
    # 单独拎出一个 1，138，138个gen对剪切位点的attention系数
    pice_attention = attention.cpu().numpy()
    # 可以做热度图，根据系数的大小，看看对剪切位点的影响
    return pice_attention
