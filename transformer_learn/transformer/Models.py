#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 上午10:01
# @Author  : Zessay

import numpy as np
import torch
import torch.nn as nn
import Constants
from .Layers import EncoderLayer, DecoderLayer

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    # 不等于PAD的为1，其余为0，扩展为3维
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''
    获取位置编码
    :param n_position: 表示总共的位置的数量
    :param d_hid: 表示编码成的维数
    :param padding_idx: 表示padding的位置
    :return:
    '''
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2*(hid_idx//2)/d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:,1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        ## 将padding位置编码置为0
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    '''
    mask掉key中属于pad的部分
    :param seq_k: 维度为 [batch_size, len_k]
    :param seq_q: 维度为 [batch_size, len_q]
    :return:
    '''
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b * len_q * len_k

    return padding_mask

def get_subsequent_mask(seq):
    '''mask掉后面词语的信息，主要用于解码阶段'''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8),
                                 diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask

# ------------------------------------------

class Encoder(nn.Module):
    '''带有自注意力的编码器模块'''
    def __init__(self, n_src_vocab, len_max_seq, d_word_vec, n_layers,
                 n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        '''
        :param n_src_vocab: 表示词表的大小
        :param len_max_seq: 表示句子的最大长度
        :param d_word_vec: 表示词向量的维度
        :param n_layers: 表示编码器的层数
        :param n_head: 表示头的数量
        :param d_k: 表示key的维度
        :param d_v: 表示value的维度
        :param d_model: 表示输入向量的维度
        :param d_inner: 表示内部隐层的维度
        :param dropout: 表示dropout的概率
        '''
        super().__init__()

        n_position = len_max_seq + 1
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True
        )

        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                                          for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        '''
        :param src_seq: 表示原始序列，维度为 [batch_size, len_s]
        :param src_pos: 表示位置序列，维度为 [batch_size, len_s]
        :param return_attns:
        :return:
        '''
        enc_slf_attn_list = []

        # 准备mask
        ## slf_attn_mask的维度为 [batch, len_s, len_s]；　non_pad_mask的维度为 [batch, len_s, 1]
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # 前向传播过程
        ## 得到前向传播的第一个张量，维度为 [batch_size, len_s, embed_size]
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            ## enc_output的维度为 [batch, len_s, embed_size]
            ## enc_slf_attn的维度为 [batch, len_s, len_s]
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


# --------------------------------------------------------

class Decoder(nn.Module):
    '''带自注意力的解码器'''
    def __init__(self, n_tgt_vocab, len_max_seq, d_word_vec, n_layers, n_head,
                 d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = len_max_seq + 1
        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD
        )

        self.position_enc =  nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True
        )

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        '''
        :param tgt_seq: 解码器的输入，维度为[batch_size, len_t]
        :param tgt_pos: 解码器的输入的位置信息，维度为 [batch_size, len_t]
        :param src_seq: 编码器的输入，维度为[batch_size, len_s]
        :param enc_output: 编码器的输出
        :param return_attns: 是否返回attn的矩阵
        :return:
        '''
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # 准备mask
        ## non_pad_mask的维度为 [batch, len_t, 1]
        non_pad_mask = get_non_pad_mask(tgt_seq)
        # 用于mask掉当前单词之后的词的矩阵
        ## slf_attn_mask_subseq的维度为 [batch, len_t, len_t]
        ## slf_attn_mask_keypad的维度为 [batch, len_t, len_t]
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        # 前向传播过程
        ## 得到的维度为 [batch, len_t, len_s]
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        ## 前向传播
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            ## dec_output维度为 [batch, len_t, embed_size]
            ## dec_slf_attn 维度为 [batch, len_t, len_t]
            ## dec_enc_attn 维度为 [batch, len_t, len_s]
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask
            )

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return dec_output

# ---------------------------------------------------------------

class Transformer(nn.Module):
    '''定义seq2seq的Transformer模型'''
    def __int__(self, n_src_vocab, n_tgt_vocab, len_max_seq,
                d_word_vec=512, d_model=512, d_inner=2048,
                n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
                tgt_emb_prj_weight_sharing=True,
                emb_src_tgt_weight_sharing=True):

        super(Transformer, self).__int__()
        self.encoder = Encoder(n_src_vocab=n_src_vocab,
                               len_max_seq=len_max_seq,
                               d_word_vec=d_word_vec,
                               d_model=d_model, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head,
                               d_k=d_k, d_v=d_v,dropout=dropout)
        self.decoder = Decoder(n_tgt_vocab=n_tgt_vocab,
                               len_max_seq=len_max_seq,
                               d_word_vec=d_word_vec,
                               d_model=d_model, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head,
                               d_k=d_k, d_v=d_v, dropout=dropout)

        # 定义从解码器输出到最终结果的投影层
        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_proj.weight)

        assert d_model == d_word_vec, "To facilitate the residual " \
                                      "connections, the dimensions of all module outputs shall be the same"

        if tgt_emb_prj_weight_sharing:
            # 如果选择共享解码器embedding层和最终投影层的参数
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1

        if emb_src_tgt_weight_sharing:
            # 是否选择共享编码器的embedding层和解码器的embedding层之间的参数
            ## 前提是源词表和目标词表数相同
            assert n_src_vocab == n_tgt_vocab, "To share word embedding table. " \
                                               "the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight


    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        '''
        :param src_seq: 源输入序列，为 [batch_size, len_s]
        :param src_pos: 源输入序列的位置信息，为[batch_size, len_s]
        :param tgt_seq: 目标输入序列，为[batch_size, len_t]
        :param tgt_pos: 目标输入序列的位置信息，为[batch_size, len_t]
        :return:
        '''
        # 这里删除最后一个位置
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        # 输出形状为[batch, len_t, tgt_vocab]
        ## 通过缩放因子，防止输出值太大
        seq_logit = self.tgt_word_proj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))