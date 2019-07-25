#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 上午10:55
# @Author  : Zessay

import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        '''
        :param q: 表示query矩阵，维度为(n*b)*lq*d_k
        :param k: 表示key矩阵，维度同上
        :param v: 表示value矩阵，维度同上
        :param mask: 表示mask矩阵，维度同上
        :return:
        '''
        # 输出形状为(n*b)*lq*lq
        attn = torch.bmm(q, k.transpose(1,2))
        attn = attn / self.temperature

        # 如果有mask则进行mask
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        # 维度为 (n*b)*lq*lq
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # 维度为 (n*b)*lq*d_v
        output = torch.bmm(attn, v)

        return output, attn
