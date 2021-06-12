# _*_ coding: utf-8 _*_
# @Time : 2021/6/6 15:03
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_data.py
# @desc :
import json

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing import sequence


class AttDataset(data.Dataset):
    def __init__(self, u_data, label):
        super().__init__()
        self.max_len = 64
        self.device = torch.device("cuda")
        # self.dataset = sequence.pad_sequences(u_data, maxlen=self.max_len)
        self.dataset = u_data
        self.labels = torch.from_numpy(label.astype(np.float64))
        # self.labels = pd.get_dummies(label).to_numpy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item1 = self.dataset[index]
        item2 = self.labels[index]
        return item1, item2
