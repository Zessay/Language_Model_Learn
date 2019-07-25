#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 下午12:08
# @Author  : Zessay

'''
定义用于组成编码器和解码器的层
'''

import torch.nn as nn
from .SubLayers import MultiHeadAttention, PositionWiseFeedForward

class EncoderLayer(nn.Module):
    '''由多头注意力和前馈网络组成的编码层'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        '''
        :param d_model: 表示输入的embedding的长度
        :param d_inner: 表示内部隐层的维度
        :param n_head: 表示头的数目
        :param d_k: 表示键和查询的维度
        :param d_v: 表示值的维度
        :param dropout: 表示dropout的概率
        '''
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        '''
        :param enc_input:
        :param non_pad_mask: 需要mask掉的位置对应元素应为0
        :param slf_attn_mask: 需要mask掉的位置对应元素应为1
        :return:
        '''
        # 这两步都需要对pad的部分进行mask
        # output的维度为 batch_size*lq*d_model, attn的维度为 (n_head*batch_size)*lq*lq
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        ## 逐元素相乘
        enc_output *= non_pad_mask
        ## 维度仍然是 batch*lq*d_model
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

# ------------------------------------------------------------

class DecoderLayer(nn.Module):
    '''由自注意力，编码器注意力，前馈网络组成的解码器'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
        :param dec_input: 表示编码器输入，维度为 [b, lq, d_model]
        :param enc_output: 解码器的输出，维度为 [b, lq, d_model]
        :param non_pad_mask: 表示对输出矩阵的pad部分进行mask的矩阵，这个矩阵是和输出矩阵按元素相乘的，所以需要mask掉的位置应为0
        :param slf_attn_mask: 表示对自注意力层进行mask的矩阵，因为解码器只能看到自己之前的内容，所以这里和encoder层的区别较大；
                              这里是用于自注意力的mask，主要包括自注意力时pad的mask以及后面单词的mask，需要mask掉的位置应为1
        :param dec_enc_attn_mask: 表示dec_enc 交互时的attn层的mask，这里同上，需要mask掉的位置应为1
        :return:
        '''
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        ## 这里表示 dec和enc的交互层，将dec_output作为query的生成矩阵，enc_output作为key和value的生成矩阵
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        ## 维度为 [batch, lq, d_model]
        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

