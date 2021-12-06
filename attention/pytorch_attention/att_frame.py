# _*_ coding: utf-8 _*_
# @Time : 2021/6/6 15:03
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_frame.py
# @desc :
import codecs
import json
import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from att_model import CNN_MULTI_BiLSTM
from dataloaders import att_dataloader
from configs import kkflod

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
    def __init__(self, batch_size, lr, max_epoch, i):
        super().__init__()
        self.model = CNN_MULTI_BiLSTM().cuda()
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.k_i = str(i)
        self.k_flod = kkflod
        if self.k_flod:
            """
            use to k fold load data
            """
            x_train = None
            y_train = None
            x_test = None
            y_test = None
            pass
        else:
            pd_train = pd.read_csv('./demo_data/train_demo.csv')
            temp = pd_train['0'].values.tolist()
            train_temp = []
            for i in temp:
                train_temp.append([eval(j) for j in i])
            x_train = torch.tensor(train_temp)
            y_train = pd_train['1'].to_numpy()
            _, x_add,  _, y_add = train_test_split(train_temp, y_train, test_size=0.1, random_state=0)
            pd_test = pd.read_csv('./demo_data/test_demo.csv')
            temp_s = pd_test['0'].values.tolist()
            test_temp = []
            for i in temp_s:
                test_temp.append([eval(j) for j in i])
            test_temp.extend(x_add)
            x_test = torch.tensor(test_temp)
            y_test = pd_test['1'].to_numpy()
            y_test = np.concatenate((y_test, y_add))
            # y_test = pd_test['label'].to_numpy()
        self.train_loader = att_dataloader(x_train, y_train, shuffle=True, batch_size=self.batch_size)
        self.test_loader = att_dataloader(x_test, y_test, shuffle=True, batch_size=self.batch_size)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_start(self):
        best_f1, best_acc, best_recall, best_A = 0, 0, 0, 0
        loss_log, acc_log, recall_log, sp_log, sn_log, mcc_log, f1_log = [], [], [], [], [], [], []
        best_pre_list, best_gold_list = [], []
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
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
            pre_list, gold_list = [], []
            old_pred_list = []
            with torch.no_grad():
                all_attention = []
                for iter_s, batch_samples in enumerate(t):
                    inputs, labels = batch_samples
                    inputs = inputs.cuda()
                    is_test = True
                    rel_out, attention = self.model(inputs, is_test)
                    # rel_out = self.model(inputs)
                    labels = labels.numpy()
                    rel_out = rel_out.to('cpu').numpy()
                    # ——————————————————————————————————
                    idx = inputs.cpu().numpy()
                    piece_attention = plot_attention(attention)
                    # for pre, gold, ids in zip(rel_out, labels, idx):  # , att, piece_attention
                    for pre, gold, ids, att in zip(rel_out, labels, idx, piece_attention):
                        pre_set = np.round(pre)
                        pre_set_1 = pre
                        old_pred_list.append(float(pre_set_1))
                        gold_set = int(gold)
                        pre_list.append(float(pre_set))
                        gold_list.append(gold_set)
                        pred_num += 1
                        gold_num += 1
                        if pre_set == gold_set:
                            correct_num += 1
                            attention_temp_dict = {'inputs': ids.tolist(), 'labels': gold, 'attention': att.tolist()}
                            all_attention.append(attention_temp_dict)
            acc_1 = correct_num/pred_num
            tn, fp, fn, tp = confusion_matrix(gold_list, pre_list).ravel()
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            sp, sn = tn / (tn + fp), tp / (tp + fn)
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            f1_score = 2 * precision * recall / (precision + recall)
            acc_log.append(acc_1)
            recall_log.append(recall)
            f1_log.append(f1_score)
            sp_log.append(sp)
            sn_log.append(sn)
            mcc_log.append(mcc)
            if best_A < acc_1:
                best_A = acc_1
            if best_f1 < f1_score:
                best_f1, best_acc, best_recall = f1_score, precision, recall
                best_pre_list = old_pred_list
                best_gold_list = gold_list
                # with codecs.open('attention_sample.json', 'w', encoding='utf-8') as f:
                #     json.dump(all_attention, f, indent=4, ensure_ascii=False)
            if acc_1 > 0.98:
                torch.save({'state_dict': self.model.state_dict()}, 'D:/Projects/DNA/dna98.pth.tar')
                print("save successful")
                break
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
            print(f'best acc{best_A}')
        print(best_f1, best_acc, best_recall,best_A)
        save_metric_dict = {
            'loss': loss_log,
            'acc': acc_log,
            'recall': recall_log,
            'f1': f1_log,
            'sp': sp_log,
            'sn': sn_log,
            'mcc': mcc_log
        }
        temp = []
        temp.extend(best_gold_list)
        temp.extend(best_pre_list)
        with codecs.open('metric.json' + self.k_i, 'w', encoding='utf-8') as f:
            json.dump(save_metric_dict, f, indent=4, ensure_ascii=False)
        with codecs.open('roc_gold.json' + self.k_i, 'w', encoding='utf-8') as f:
            json.dump(best_gold_list, f, indent=4, ensure_ascii=False)
        with codecs.open('roc_pre.json' + self.k_i, 'w', encoding='utf-8') as f:
            json.dump(best_pre_list, f, indent=4, ensure_ascii=False)


def plot_attention(attention):
    attention = attention[:, :, 70:72, :]
    attention = torch.mean(attention, 1)
    pice_attention = attention.cpu().numpy()
    return pice_attention


def plot_roc(labels, predict_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
