# _*_ coding: utf-8 _*_
# @Time : 2021/12/6 15:16
# @Author : 张宝宇
# @Version：V 1.0
# @File : dataloaders.py
# @desc :
import torch.utils.data as data

from att_data import AttDataset


def att_dataloader(u_data=None, label=None, batch_size=None, shuffle=None, num_workers=0):
    dataset = AttDataset(u_data, label)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers
                                  )
    return data_loader
