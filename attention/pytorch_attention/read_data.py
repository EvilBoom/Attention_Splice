# _*_ coding: utf-8 _*_
# @Time : 2021/6/9 11:04
# @Author : 张宝宇
# @Version：V 0.0
# @File : read_data.py
# @desc :
import codecs
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

s_dict = {'A': '0', 'C': '1', 'G': '2', 'T': '3', 'H': '4'}


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
    with open('../../datasets/sds_pos.txt') as f:
        data = f.readlines()
        origin_str = ''
        for item in data:
            origin_str += item.replace('\n', '')
        origin_list = origin_str.split('>')
        origin_list = origin_list[1:]
        origin_list = [i + '\n' for i in origin_list]
    with open('sds_test.txt', 'w') as f:
        f.writelines(origin_list)


def pro_precess():
    with open('sds_neg_test.txt') as f:
        data = f.readlines()
        neg_data_list = []
        for i in data:
            neg_data_list.append(i.replace('\n', ''))
    with open('sds_test.txt') as f:
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
    # new_data_list = []
    # for i in data_list:
    #     i = i.replace('\n', '')
    #     temp = []
    #     for ins in range(0, len(i), 1):
    #         if ins == 138:
    #             break
    #         item = i[ins: ins + 3]
    #         temp.append(temp_dict[item])
    #     new_data_list.append(temp)
    for ids, i in enumerate(data_list):
        temp = ''.join([s_dict[j] for j in i])
        data_list[ids] = temp
    d = {'sentence': data_list, 'label': label_list}
    pds = pd.DataFrame(data=d)
    pds = shuffle(pds)
    nds = pds.to_numpy()
    pds2_train = pd.DataFrame(nds[:5000])
    pds2_test = pd.DataFrame(nds[5000:])
    pds2_train.to_csv('d_num_train.csv')
    pds2_test.to_csv('d_num_test.csv')


def hs3d_data():
    with open(r'D:\Projects\DNA\datasets\HS3D Datasets\IE_true.seq.txt', 'r') as f:
        data = f.readlines()
    regexs_1 = r'.*?: (.*?)$'
    pos_sample = []
    for item in tqdm(data):
        seq = re.match(regexs_1, item)
        if seq:
            seq_data = seq.group(1)
            seq_data = ''.join([s_dict[j] for j in seq_data])
        else:
            print(1)
            continue
        pos_sample.append(seq_data)
    pos_label = [1.0] * len(pos_sample)
    save_data = {'seq': [], 'label': []}
    save_data['seq'].extend(pos_sample)
    save_data['label'].extend(pos_label)
    with open(r'D:\Projects\DNA\datasets\HS3D Datasets\IE_false.seq.txt', 'r') as f:
        data2 = f.readlines()
    neg_sample = []
    for item in tqdm(data2):
        seq = re.match(regexs_1, item)
        if seq:
            seq_data = seq.group(1)
            seq_data = ''.join([s_dict[j] for j in seq_data])
        else:
            print(1)
            continue
        neg_sample.append(seq_data)
    neg_label = [0.0] * len(neg_sample)
    save_data['seq'].extend(neg_sample)
    save_data['label'].extend(neg_label)
    # pd_data = pd.DataFrame(save_data).to_numpy()
    x = np.array(save_data['seq'])
    y = np.array(save_data['label'])
    pd.DataFrame(save_data).to_csv('./sas_Hs3_test.csv')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True)
    train_data = {'seq': [], 'label': []}
    test_data = {'seq': [], 'label': []}
    train_data['seq'].extend(X_train)
    test_data['seq'].extend(X_test)
    train_data['label'].extend(y_train)
    test_data['label'].extend(y_test)
    train_data_pd = pd.DataFrame(train_data)
    test_data_pd = pd.DataFrame(test_data)
    train_data_pd.to_csv('./num_train_sas_Hs3.csv')
    test_data_pd.to_csv('./num_test_sas_Hs3.csv')


if __name__ == '__main__':
    # process_data()
    # pro_precess()
    # triple_dict2json()
    # train = pd.read_csv('d_train.csv')
    # triple_get_data()
    hs3d_data()
    print(1)
