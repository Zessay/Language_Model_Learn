#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 下午5:09
# @Author  : Zessay

'''
定义束搜索的模块
'''

import torch
import numpy as np
import Constants

class Beam():
    '''束搜索模块'''
    def __init__(self, size, device=False):
        '''
        :param size: 表示束的大小
        :param device:
        '''
        self.size = size
        self._done = False

        # 保存每一个束上翻译结果的分数
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # 每一个时间点的回溯位置
        self.prev_ks = []

        # 每一个时间步的输出单词
        self.next_ys = [torch.full((size,), Constants.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = Constants.BOS

    def get_current_state(self):
        '''获取当前时间步的输出'''
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        '''获取当前时间步上一个时间步的结果'''
        return self.prev_ks[-1]

    # 相当于一个get方法
    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        '''
        更新束的状态并检查是否完成
        :param word_prob: 表示当前所有束的下一个词概率，形状为 [n_bm, tgt_vocab]
        '''
        num_words = word_prob.size(1)

        # 将之前的分数求和
        if len(self.prev_ks)>0:
            ## 如果不是第一步，则把对应位置上一个束的概率加上去，因为是对数概率，所以是加
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            # 如果是第一步
            ## 这里之所以是word_prob[0]
            ## 是因为上面的初始状态next_ys只有第0个位置是BOS
            ## 所以直接取该束上下一个单词概率即可
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        # self.size表示topk中的k，0表示排序的维度
        #best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 第一次排序
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 第二次排序

        ## 所以self.socres记录的是上一次最佳n_bm的概率值
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # BestScoresId 展开为 beam*word的一维数组，所以需要计算每个分数来自哪一个单词和束
        ## 计算前一个所在的beam，维度为[n_beam]
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        ## 得到当前所表示的单词的id，　维度为[n_beam]
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # 当束的最末尾是'EOS'的时候停止搜索
        ## -1表示最近一次返回的ID，0表示概率最大的单词，如果最近返回的概率最大的单词为EOS，表示翻译完成
        ## all_scores保存最后翻译完成时，各个束最终概率
        if self.next_ys[-1][0].item() == Constants.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        '''对分数进行排序'''
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        '''获取束中最好的分数的路径'''
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        '''对当前的时间步获取解码序列'''
        ## 如果是第一次调用，则返回初始值
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)
        return dec_seq

    def get_hypothesis(self, k):
        '''回溯找到全概率'''
        hyp = []
        ## 从后往前回溯找到位于该束上的单词
        ## prev_ks表示该位置上一个束的索引
        ## next_ys记录当前位置对应束上的单词
        for j in range(len(self.prev_ks)-1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
        # 转换为列表的形式返回
        return list(map(lambda x: x.item(), hyp[::-1]))