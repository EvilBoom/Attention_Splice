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
from torch.autograd import Variable


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=True):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
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


class CNN_MULTI_BiLSTM(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=768, hidden_dim=256, output_dim=1, n_layers=4,
                 bidirectional=True, dropout=0.1):
        super(CNN_MULTI_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.multi = MultiHeadAttention(hidden_dim * 2, hidden_dim * 2, 16)
        self.drop = nn.Dropout(0.1)

        self.is_training = True
        self.window_sizes = [3, 4, 8, 9]
        self.cnn_out = 100
        self.max_text_len = 140
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=hidden_dim * 2,
                                    out_channels=self.cnn_out,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=self.cnn_out),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.max_text_len - h + 1))
            for h in self.window_sizes
        ])
        self.fc = nn.Linear(in_features=self.cnn_out * len(self.window_sizes),
                            out_features=1)

    def forward(self, text, is_test):
        bl_batch_size = text.size(0)
        embedded = self.dropout(self.embedding(text))
        embedded = embedded.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(2 * self.n_layers, bl_batch_size, self.hidden_dim).cuda())
        c_0 = Variable(torch.zeros(2 * self.n_layers, bl_batch_size, self.hidden_dim).cuda())
        output, (hidden, final_cell_state) = self.rnn(embedded, (h_0, c_0))
        output = output.permute(1, 0, 2)
        att_out, attention = self.multi(output)
        att_out = F.dropout(att_out, p=self.dropout_rate)
        embed_x = att_out.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out).squeeze(1)
        out = out.sigmoid()
        if is_test:
            return out, attention
        else:
            return out
