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
        self.embed_dim = 10
        self.embed = nn.Embedding(4, self.embed_dim)
        # self.att = nn.MultiheadAttention(128, 4)
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.lin_q = nn.Linear(self.embed_dim, 100)
        self.lin_k = nn.Linear(self.embed_dim, 100)
        self.lin_v = nn.Linear(self.embed_dim, 100)
        self.drop = nn.Dropout(0.01)
        self.lin1 = nn.Linear(self.embed_dim, 1)
        self.lin2 = nn.Linear(64, 1)
        self.multi = MultiHeadAttention(self.embed_dim, self.embed_dim, 2)

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
        att_out = torch.mean(att_out, 1)
        pool_out = self.drop(att_out)
        ret_out = self.lin1(pool_out).squeeze(1)
        # ret_out = self.lin2(ret_out).squeeze(1)

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


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = 0.1
        self.embedding_dim = 100
        self.window_sizes = [3, 4, 5, 6]
        self.max_text_len = 140
        self.embedding = nn.Embedding(num_embeddings=4,
                                      embedding_dim=self.embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim,
                                    out_channels=100,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.max_text_len - h + 1))
            for h in self.window_sizes
        ])
        self.fc = nn.Linear(in_features=100 * len(self.window_sizes),
                            out_features=1)

    def forward(self, x):
        embed_x = self.embedding(x)
        # print('embed size 1',embed_x.size())  # 32*140*256
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
        # print('embed size 2',embed_x.size())  # 32*256*140
        out = [conv(embed_x) for conv in self.convs]  # out[i]:batch_size x feature_size*1
        # for o in out:
        #    print('o',o.size())  # 32*100*1
        out = torch.cat(out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        # print(out.size(1)) # 32*400*1
        out = out.view(-1, out.size(1))
        # print(out.size())  # 32*400
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out).squeeze(1)  # 32 * 1
        out = out.sigmoid()
        return out

