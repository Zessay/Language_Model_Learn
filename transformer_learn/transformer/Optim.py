#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 上午9:46
# @Author  : Zessay

'''优化器的封装类'''

import numpy as np

class ScheduledOptim:
    '''学习率随时间变化的优化器类'''
    def __init__(self, optimizer, d_model, n_warmup_steps):
        '''
        :param optimizer: 表示内部的优化器
        :param d_model: 表示embedding的维度
        :param n_warmup_steps: 表示前期增大steps的阶段
        '''
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        '''使用内部优化器逐步更新'''
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        '''使用内部优化器清除梯度'''
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        '''对学习率进行衰减或增大'''
        # 这里说明当　current_steps < warmup_steps的时候，　学习率是逐渐增大的
        # 当current_steps > warmup_steps的时候，学习率开始减小
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5)*self.n_current_steps
        ])

    def _update_learning_rate(self):
        '''更新梯度'''
        self.n_current_steps += 1
        lr = self.init_lr*self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr