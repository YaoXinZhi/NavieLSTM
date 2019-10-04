#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 04/10/2019 14:05 
@Author: XinZhi Yao 
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parameter as Parameter

class NaiveLSTM(nn.Module):
    """
    Naive LSTM like nn.LSTM
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(NaiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.w_ii = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hi = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hi = torch.nn.Parameter(torch.Tensor(hidden_size, 1))

        # forget gate
        self.w_if = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hf = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hf = torch.nn.Parameter(torch.Tensor(hidden_size, 1))

        # output gate
        self.w_io = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_ho = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_ho = torch.nn.Parameter(torch.Tensor(hidden_size, 1))

        # cell weight and bias
        self.w_ig = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hg = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hg = torch.nn.Parameter(torch.Tensor(hidden_size, 1))

        self.reset_weight()

    def reset_weight(self):
        """
        reset weight
        :return: None
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, state_t):
        """
        Forward
        :param inputs: [batch_size, seq_size, input_size]
        :param state_t: ([batch_size, 1, hidden_size], [batch_size, 1, hidden_size])
        :return: hidden_seq [batch_size, seq_len, hidden_size]
                (h_next_t, c_next_t) ([batch_size, 1, 20], [batch_size, 1, 20])
        """

        # batch_first = True
        batch_size, seq_size, _ = inputs.size()

        if state_t is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state_t
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []

        for t in range(seq_size):
            x = inputs[:, t, :].t()
            # input gate
            i = torch.sigmoid(self.w_ii@x + self.b_ii + self.w_hi@h_t + self.b_hi)

            # forget gate
            f = torch.sigmoid(self.w_if@x + self.b_if + self.w_hf@h_t + self.b_hf)

            # cell state
            g = torch.tanh(self.w_ig@x + self.b_ig + self.w_hg@h_t + self.b_hg)

            # output gate
            o = torch.sigmoid(self.w_io@x + self.b_io + self.w_ho@h_t + self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=1)
        # here h_next_t and c_next_t means hidden state and cell
        # state in last moment
        return hidden_seq, (h_next_t, c_next_t)

def reset_weights(model):
    """
    reset weight
    :param model: the model you want to reset weight
    :return: None
    """
    for weight in model.parameters():
        init.constant_(weight, 0.5)

if __name__ == '__main__':

    #  fake data (batch_first = True)
    inputs = torch.ones(1, 1, 10)  # [batch_size, seq_size, input_size]
    h0 = torch.ones(1, 1, 20)  # [batch_size, seq_size, hidden_size]
    c0 = torch.ones(1, 1, 20)  # [batch_size, seq_size, hidden_size]
    print('h0 shape: {0}'.format(h0.shape))
    print('c0 shape: {0}'.format(c0.shape))
    print('inputs shape: {0}'.format(inputs.shape))

    # initialize naive lstm with inputs_size=10, hidden_size=20
    naive_lstm = NaiveLSTM(input_size=10, hidden_size=20)
    reset_weights(naive_lstm)

    # test model
    output, (hn1, cn1) = naive_lstm(inputs, (h0, c0))
    print('shape --> hn1: {0} | cn1: {1} | output: {2}'.format(hn1.shape, cn1.shape, output.shape))
