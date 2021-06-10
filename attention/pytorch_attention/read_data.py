# _*_ coding: utf-8 _*_
# @Time : 2021/6/9 11:04
# @Author : 张宝宇
# @Version：V 0.0
# @File : read_data.py
# @desc :
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

s_dict = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}


def process_data():
    with open('../../datasets/sas_pos.txt') as f:
        data = f.readlines()
        pos_data_list = []
        origin_str = ''
        for item in data:
            origin_str += item.replace('\n', '')
        origin_list = origin_str.split('>')
        origin_list = origin_list[1:]
        origin_list = [i + '\n' for i in origin_list]
    with open('test.text', 'w') as f:
        f.writelines(origin_list)


def pro_precess():
    with open('../../datasets/sas_neg.txt') as f:
        data = f.readlines()
        neg_data_list = []
        for index, item in enumerate(data):
            if index == 0:
                old_str = ''
                continue
            elif index % 4 == 0:
                neg_data_list.append(old_str)
                old_str = ''
            else:
                old_str += item.replace('\n', '')
        a = set()
        for ins, i in enumerate(neg_data_list):
            a.add(len(i))
            if len(i) == 139:
                print(ins)
    with open('test.text') as f:
        data = f.readlines()
        pos_data_list = []
        for i in data:
            pos_data_list.append(i.replace('\n', ''))
    data_list = []
    label_list = []
    pos = np.ones(len(pos_data_list))
    neg = np.zeros(len(neg_data_list))
    data_list.extend(pos_data_list)
    data_list.extend(neg_data_list)
    label_list.extend(neg)
    label_list.extend(pos)
    new_data_list = []
    for i in data_list:
        mod_if = []
        mod_if[:0] = i
        for ins, j in enumerate(mod_if):
            mod_if[ins] = s_dict[j]
        mod_if = ''.join(mod_if)
        new_data_list.append(mod_if)
    d = {'sentence': new_data_list, 'label': label_list}
    pds = pd.DataFrame(data=d)
    pds = shuffle(pds)
    nds = pds.to_numpy()
    pds2_train = pd.DataFrame(nds[:5000])
    pds2_test = pd.DataFrame(nds[5000:])
    pds2_train.to_csv('num_train.csv')
    pds2_test.to_csv('num_test.csv')


if __name__ == '__main__':
    # process_data()
    pro_precess()
    train = pd.read_csv('d_train.csv')
    print(1)
