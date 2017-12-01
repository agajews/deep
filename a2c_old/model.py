"""
File adapted from https://github.com/atgambardella/pytorch-es/blob/master/model.py
"""

import torch.nn as nn
import torch.nn.functional as F


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class SeluModel(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SeluModel, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        self.actor_linear = nn.Linear(256, num_outputs)
        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = selu(self.conv1(x))
        x = selu(self.conv2(x))
        x = selu(self.conv3(x))
        x = selu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.actor_linear(x), (hx, cx)
