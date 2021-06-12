# _*_ coding: utf-8 _*_
# @Time : 2021/6/9 11:04
# @Author : 张宝宇
# @Version：V 0.0
# @File : read_data.py
# @desc :
import codecs
import json

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

s_dict = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}


def triple_get_data():
    with open('../../datasets/sas_pos.txt') as f:
        data = f.readlines()
        pos_data_list = []
        origin_str = ''
        for item in data:
            origin_str += item.replace('\n', '')
    with open('temp.txt', 'w') as f:
        f.writelines(origin_str)


def triple_dict2json():
    with open('temp.txt') as f:
        origin_str = f.readlines()[0]
    gen_sets = set()
    for i in range(0, len(origin_str), 3):
        print(origin_str[i: i + 3])
        gen_sets.add(origin_str[i: i + 3])
    gen_dict = {}
    for ins, j in enumerate(gen_sets):
        gen_dict[j] = ins
    with codecs.open('dict.json', 'w', encoding='utf-8') as f:
        json.dump(gen_dict, f, indent=4, ensure_ascii=False)
    with open('dict.json') as f:
        temp_dict = json.load(f)
    print(1)


def process_data():
    # with open('../../datasets/sas_pos.txt') as f:
    with open('../../datasets/sas_neg.txt') as f:
        data = f.readlines()
        origin_str = ''
        for item in data:
            origin_str += item.replace('\n', '')
        origin_list = origin_str.split('>')
        origin_list = origin_list[1:]
        origin_list = [i + '\n' for i in origin_list]
    with open('sas_neg_test.txt', 'w') as f:
        f.writelines(origin_list)


def pro_precess():
    with open('sas_neg_test.txt') as f:
        data = f.readlines()
        neg_data_list = []
        for i in data:
            neg_data_list.append(i.replace('\n', ''))
    with open('sas_test.txt') as f:
        data = f.readlines()
        pos_data_list = []
        for i in data:
            pos_data_list.append(i.replace('\n', ''))
    with open('dict.json') as f:
        temp_dict = json.load(f)
    data_list, label_list = [], []
    pos, neg = np.ones(len(pos_data_list)), np.zeros(len(neg_data_list))
    data_list.extend(pos_data_list)
    data_list.extend(neg_data_list)
    label_list.extend(neg)
    label_list.extend(pos)
    new_data_list = []
    for i in data_list:
        i = i.replace('\n', '')
        temp = []
        for ins in range(0, len(i), 1):
            if ins == 138:
                break
            item = i[ins: ins + 3]
            temp.append(temp_dict[item])
        new_data_list.append(temp)
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
    # triple_dict2json()
    # train = pd.read_csv('d_train.csv')
    # triple_get_data()

    print(1)
