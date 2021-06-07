# _*_ coding: utf-8 _*_
# @Time : 2021/6/6 15:01
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_model.py
# @desc :
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchAttention(nn.Module):
    def __init__(self):
        super(TorchAttention, self).__init__()
        self.embed = nn.Embedding(20000, 128)
        # self.att = nn.MultiheadAttention(128, 4)
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.lin_q = nn.Linear(128, 100)
        self.lin_k = nn.Linear(128, 100)
        self.lin_v = nn.Linear(128, 100)
        self.drop = nn.Dropout(0.01)
        self.lin1 = nn.Linear(128, 2)

    def forward(self, x):
        # x shape 32,64
        out = self.embed(x)  # out shape 32,64,128
        # 直接使用多头注意力机制
        # t_out = out.transpose(0, 1)
        # t_att_out, att_out_weights = self.att(t_out, t_out, t_out)
        # pool_out = self.pool(att_out)
        # att_out = t_att_out.transpose(0, 1)
        # ————————————————————————————————————————————————————————————————————————————————————————————————————————————
        # self-attention
        queries = self.lin_q(out)
        keys = self.lin_k(out)
        value = self.lin_v(out)  # 32,64,100
        att_scores = queries @ keys.transpose(1, 2)
        att_score_soft = F.softmax(att_scores, dim=-1)  # 32,64,64
        att_out = value[:, :, None] * att_score_soft.transpose(1, 2)[:, :, :, None]

        pool_out = torch.mean(att_out, 1)
        pool_out = self.drop(pool_out)
        ret_out = self.lin1(pool_out)
        # ret_out = self.soft(ret_out)
        ret_out = ret_out.sigmoid()
        return ret_out
