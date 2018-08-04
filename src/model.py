#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 num_2d_coords,
                 num_3d_coords,
                 linear_size,
                 num_stage,
                 p_dropout,
                 predict_scale,
                 scale_range,
                 unnorm_op,
                 unnorm_init):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.scale_range = scale_range
        self.unnorm_op = unnorm_op
        self.predict_scale = predict_scale

        # 2d joints
        self.input_size =  num_2d_coords
        # 3d joints
        self.output_size = num_3d_coords

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        # weights that predict the scale of the image
        self.ws = nn.Linear(self.linear_size, 1)
        # sigmoid that makes sure the resulting scale is positive
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        # self.mult = nn.Parameter(torch.ones(1)*unnorm_init)
        self.mult = nn.Parameter(torch.ones(num_3d_coords)*unnorm_init)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        out = self.w2(y)

        # apply the unnormalization parameters to the output
        if self.unnorm_op:
            out = out*self.mult

        # predict the scale that will multiply the poses
        scale = self.scale_range * self.sigmoid(self.ws(y))
        return out, scale
