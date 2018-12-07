import torch
import torch.nn as nn
from torch.autograd import Function

class SignFunction(Function):
    '''Adds quantization noise during training, not testing'''

    def __init__(self):
        super(Sign, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] =  1
            x[(1 - input) / 2 >  prob] = -1
            return x
        else:
            return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Sign(nn.Module):
    def __init__(self):
        super().__init__()
        self.sign = SignFunction

    def forward(self, x):
        return self.sign.apply(x, self.training)
