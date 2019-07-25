#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 上午10:39
# @Author  : Zessay

'''定义编码器和解码器的子层'''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .Modules import ScaledDotProductAttention

# 定义多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        '''
        :param n_head: 表示头的数量
        :param d_model: 表示embedding的维度
        :param d_k:  表示k的维度
        :param d_v:  表示v的维度
        :param dropout: 表示对embedding进行dropout的概率
        '''
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head*d_k)
        self.w_ks = nn.Linear(d_model, n_head*d_k)
        self.w_vs = nn.Linear(d_model, n_head*d_v)
        # 对几个线性层的权重进行初始化
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))  # 均值和0，方差为维度倒数的平方根
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()    # sz_b表示batch的大小, len_q表示序列的长度
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # 维度为(n*b) * lq * dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)  # 表示沿着第一维重复n_head次，其他两个维度保持不变，即为 (n*b) * .. * ..
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b*lq*(n*dv)

        output = self.dropout(self.fc(output))  # 这里保证最后的输出形式为 b*lq*d_model，便于之后的残差处理
        output = self.layer_norm(output+residual)

        return output, attn


class PositionWiseFeedForward(nn.Module):
    '''两层的前向传输层'''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x: 经过多头注意力层的输出，维度为 [batch, lq, d_model]
        :return:
        '''
        residual = x
        output = x.transpose(1,2)  # torch的conv要求channel在前，即输出的形式为(N, channel, L)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        ## 输出维度为 batch*lq*d_model
        output = self.layer_norm(output + residual)
        return output

