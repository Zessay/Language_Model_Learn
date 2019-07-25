#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 下午3:06
# @Author  : Zessay

'''该模块会采用束搜索的方法实现文本的生成'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Models import Transformer
from .Beam import Beam

class Translator(object):
    '''加载训练好的模型并使用束搜索的方法进行解码'''
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")

        # 加载训练好的模型
        checkpoint = torch.load(opt.model)
        model_opt = checkpoint["settings"]
        self.model_opt = model_opt

        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
            d_k = model_opt.d_k, d_v=model_opt.d_v,
            d_model=model_opt.d_model, d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid, n_layers=model_opt.n_layers,
            n_head=model_opt.n_head, dropout=model_opt.dropout
        )

        model.load_state_dict(checkpoint["model"])
        print("[Info] Trained model state loaded")

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def translate_batch(self, src_seq, src_pos):
        '''在一个Batch内进行翻译'''
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            '''
            :param inst_idx_list: list型，表示batch大小的数字顺序列表
            :return:
            '''
            # inst_idx表示语句的ID, tensor_position表示在当前张量中所处的位置
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            '''返回和活跃的instance对应的张量部分'''
            ## 对于src_seq，这里的d_hs应该是 [len_s]；对于src_enc，应该是[len_s, embed_size]
            ## n_curr_active_inst记录为完成解码的instances数量
            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst*n_bm, *d_hs)

            ## 下面的3步完成形状转换，选择未完成解码的instances对应的索引
            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # 收集仍然活跃的句子，如果已完成则不再解码
            ## 记录上一次仍未完成解码的instance数量
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            ## 根据闭包的特性，调用该函数之前的代码中定义的变量，这里可以直接使用，比如　n_bm
            ## 对下面3个张量的内容进行更新
            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            ### 这里开始的时候纠结了比较久的时间，后面发现是没有仔细看清active_inst_idx和active_inst_idx_list
            ### 注意： active_inst_idx_list里面的值一直是是[0, n_inst)这个范围的
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            '''
            解码及更新beam的状态，然后返回活跃的beam的idx
            :param inst_dec_beams: list型，其中每一个变量都是Beam对象
            :param len_dec_seq: int型，表示当前解码的长度，也就是解码到第几个单词
            :param src_seq: 将原始的src_seq进行了扩展，形状为[batch*n_bm, len_s]
            :param enc_output: 将原始的enc_output进行了扩展，形状为 [batch*n_bm, len_s, embed_size]
            :param inst_idx_to_position_map: 记录实例索引到当前位置的映射
            :param n_bm: 表示束的数量
            :return:
            '''
            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                ## 获取还未完成的instance的当前解码状态
                ### get_current_state返回的是[n_bm, len_dec_seq]的解码结果，里面的元素是对应单词
                ### 所以最终返回值的大小为 [n_active_inst*n_bm, len_dec_seq]
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                ## 输出的dec_partial_pos: [n_active_inst*n_bm, len_dec_seq]
                ### 每个元素表示解码的序列长度序数
                dec_partial_pos = torch.arange(1, len_dec_seq+1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst*n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
                '''
                :param dec_seq: 解码输出序列，维度为 [n_active_inst*n_beam, len_dec_seq]
                :param dec_pos: 未解码完的序列当前解码到的位置， [n_active_inst*n_bm, len_dec_seq]
                '''
                # 这里的含义其实就是，把每一个active instance所包含的beam当成一个独立的语句，给到解码器进行解码
                ## 输出 [n_active_inst*n_beam, len_dec_seq, tgt_embed]
                ## 因为只是要预测最后一个词，所以只需要最后一个step的结果
                ## 于是得到对于每一个beam对应的下一个单词的概率word_prob，形状为 [n_active_inst, n_bm, n_tgt_vocab]
                dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # 选择最后一个step的结果
                word_prob = F.log_softmax(self.model.tgt_word_proj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collective_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                ## 记录仍然未完成解码的instance对应的index
                active_inst_idx_list = []
                ## 对于每一个instance
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            # 记录当前还未完成解码的instance数
            n_active_inst = len(inst_idx_to_position_map)

            ## dec_seq: [n_active_inst*n_beam, len_dec_seq]; dec_pos: [n_active_inst*n_bm, len_dec_seq]
            ## word_prob: [n_active_inst, n_bm, n_tgt_vocab]
            ## 对于这里的dec_seq,
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            # 基于预测的单词概率更新束，并获取未完成的instances
            ## 返回值是list型，表示未完成解码的instances在上一次中的索引
            active_inst_idx_list = collective_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)
            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            '''
            :param inst_dec_beams: 表示每个instance对应的Beam对象，[batch]
            :param n_best: 表示想要返回前多少个最佳结果
            :return:
            '''
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                ## get_hypothesis返回的是最佳结果的序列，是list型，其中每个元素是单词对应的索引
                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]

            return all_hyp, all_scores

        with torch.no_grad():
            # 编码阶段
            ## src_seq: [batch, len_s]; src_pos: [batch, len_s]
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            ## src_enc: [batch, len_s, embed_size]
            src_enc, *_ = self.model.encoder(src_seq, src_pos)

            ## 为了进行束搜索，重复数据
            ### 定义束的大小
            n_bm = self.opt.beam_size
            n_inst, len_s, d_h = src_enc.size()
            ### src_seq: [batch*n_bm, len_s]; src_enc: [batch*n_bm, len_s, embed_size]
            src_seq = src_seq.repeat(1, n_bm).view(n_inst*n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst*n_bm, len_s, d_h)

            ## 准备束
            ### 大小为 [batch]
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            ## 用来记录仍然没有完成解码的instance，大小为[batch]
            ### 转换之后的inst_idx_to_position_map是dict型，其中键是对应的sentence，值的初始状态和键相同
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # 解码阶段
            ## lec_dec_seq 从1开始
            for len_dec_seq in range(1, self.model_opt.max_token_seq_len+1):
                ## 返回仍然未完成解码的instances
                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)
                ## 如果不存在未完成解码的instance则结束
                if not active_inst_idx_list:
                    break

                # 这里对inst_idx_to_position_map的更新有疑问
                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        ## 返回序列以及对应的最终概率
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.opt.n_best)

        return batch_hyp, batch_scores