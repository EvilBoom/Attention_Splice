# _*_ coding: utf-8 _*_
# @Time : 2021/6/15 9:06
# @Author : 张宝宇
# @Version：V 0.0
# @File : plt_plot.py
# @desc :
import json

import matplotlib.pyplot as plt


def plt_loss():
    with open('./bi_lstm_log/bi_mul_train_loss.json', 'r') as f:
        bi_lstm_log = json.load(f)
    with open('./bi_lstm_multi_log/bi_mul_train_loss.json', 'r') as f:
        bi_lstm_mul_log = json.load(f)
    with open('./TextCNN/bi_mul_train_loss.json', 'r') as f:
        text_att_log = json.load(f)
    with open('./muti_attention/bi_mul_train_loss.json', 'r') as f:
        muti_log = json.load(f)
    y = range(200)
    plt.figure()
    plt.plot(y, bi_lstm_log, 'r--', label='BiLSTM')
    plt.plot(y, bi_lstm_mul_log, 'g--', label='BiLSTM_MULTI')
    plt.plot(y, text_att_log, 'b--', label='Text_CNN')
    plt.plot(y, muti_log, 'y--', label='MULTI_Attention')

    plt.legend()
    plt.savefig('loss_compare.jpg')
    plt.show()


def plot_acc():
    with open('./bi_lstm_log/bi_mul_train_acc.json', 'r') as f:
        bi_lstm_log = json.load(f)
    with open('./bi_lstm_multi_log/bi_mul_train_acc.json', 'r') as f:
        bi_lstm_mul_log = json.load(f)
    with open('./TextCNN/bi_mul_train_acc.json', 'r') as f:
        text_att_log = json.load(f)
    with open('./muti_attention/bi_mul_train_acc.json', 'r') as f:
        muti_log = json.load(f)
    y = range(200)
    plt.figure()
    plt.plot(y, bi_lstm_log, 'r--', label='BiLSTM')
    plt.plot(y, bi_lstm_mul_log, 'g--', label='BiLSTM_MULTI')
    plt.plot(y, text_att_log, 'b--', label='Text_CNN')
    plt.plot(y, muti_log, 'y--', label='MULTI_Attention')
    plt.legend()
    plt.savefig('acc_compare.jpg')
    plt.show()


def plot_recall():
    with open('./bi_lstm_log/bi_mul_train_recall.json', 'r') as f:
        bi_lstm_log = json.load(f)
    with open('./bi_lstm_multi_log/bi_mul_train_recall.json', 'r') as f:
        bi_lstm_mul_log = json.load(f)
    y = range(200)
    plt.figure()
    plt.plot(y, bi_lstm_log, 'r--', label='BiLSTM')
    plt.plot(y, bi_lstm_mul_log, 'g--', label='BiLSTM_MULTI')
    plt.legend()
    plt.savefig('recall_compare.jpg')
    plt.show()


if __name__ == '__main__':
    plt_loss()
    plot_acc()
    # plot_recall()
    pass
