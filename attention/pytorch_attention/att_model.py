# _*_ coding: utf-8 _*_
# @Time : 2021/6/6 15:01
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_model.py
# @desc :
import math

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

        self.multi = MultiHeadAttention(128, 128, 3)

    def forward(self, x):
        # x shape 32,64
        out = self.embed(x)  # out shape 32,64,128
        # 使用多头注意力机制
        # t_out = out.transpose(0, 1)
        # t_att_out, att_out_weights = self.att(t_out, t_out, t_out)
        # pool_out = self.pool(att_out)
        # att_out = t_att_out.transpose(0, 1)
        # ————————————————————————————————————————————————————————————————————————————————————————————————————————————
        # self-attention
        # queries = self.lin_q(out)
        # keys = self.lin_k(out)
        # value = self.lin_v(out)  # 32,64,100
        # att_scores = queries @ keys.transpose(1, 2)
        # att_score_soft = F.softmax(att_scores, dim=-1)  # 32,64,64
        # att_out = value[:, :, None] * att_score_soft.transpose(1, 2)[:, :, :, None]
        # ———————————————————————————————————————————————————————————————————————————————
        att_out = self.multi(out)
        pool_out = torch.mean(att_out, 1)
        pool_out = self.drop(pool_out)
        ret_out = self.lin1(pool_out)
        # ret_out = self.soft(ret_out)
        ret_out = ret_out.sigmoid()
        return ret_out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_log_its = torch.matmul(q, k.transpose(-2, -1))
    attn_log_its = attn_log_its / math.sqrt(d_k)
    if mask is not None:
        attn_log_its = attn_log_its.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_log_its, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention
