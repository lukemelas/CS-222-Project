import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ConvLSTMCell, Sign

class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, 
                              bias=False)
        self.rnn1 = ConvLSTMCell(64, 256, kernel_size=3, stride=2, padding=1, 
                                 hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTMCell(256, 512, kernel_size=3, stride=2, padding=1, 
                                 hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTMCell(512, 512, kernel_size=3, stride=2, padding=1, 
                                 hidden_kernel_size=1, bias=False)

    def forward(self, input, hs):
        h1, h2, h3 = hs

        x = self.conv(input)

        h1 = self.rnn1(x, h1)
        x = h1[0]
        h2 = self.rnn2(x, h2)
        x = h2[0]
        h3 = self.rnn3(x, h3)
        x = h3[0]
        
        return x, [h1, h2, h3]

    def create_zeros(self, dims, gpu, grad):
        tmp = (torch.zeros(*dims, requires_grad=grad), 
               torch.zeros(*dims, requires_grad=grad))
        tmp = (tmp[0].cuda(), tmp[1].cuda()) if gpu else tmp
        return tmp
        
    def create_hidden(self, batch_size, gpu=True, grad=True):
        h1 = self.create_zeros([batch_size, 256, 8, 8], gpu, grad)
        h2 = self.create_zeros([batch_size, 512, 4, 4], gpu, grad)
        h3 = self.create_zeros([batch_size, 512, 2, 2], gpu, grad)
        return [h1, h2, h3]

class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = torch.tanh(feat)
        return self.sign(x)

class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()

        self.conv1 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.rnn1 = ConvLSTMCell(512, 512, kernel_size=3, stride=1, padding=1,
                                 hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTMCell(128, 512, kernel_size=3, stride=1, padding=1,
                                 hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTMCell(128, 256, kernel_size=3, stride=1, padding=1,
                                 hidden_kernel_size=3, bias=False)
        self.rnn4 = ConvLSTMCell(64, 128, kernel_size=3, stride=1, padding=1,
                                 hidden_kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, 
                               bias=False)

    def forward(self, input, hs):
        h1, h2, h3, h4 = hs

        x = self.conv1(input)

        h1 = self.rnn1(x, h1)
        x = h1[0]
        x = F.pixel_shuffle(x, 2)

        h2 = self.rnn2(x, h2)
        x = h2[0]
        x = F.pixel_shuffle(x, 2)

        h3 = self.rnn3(x, h3)
        x = h3[0]
        x = F.pixel_shuffle(x, 2)

        h4 = self.rnn4(x, h4)
        x = h4[0]
        x = F.pixel_shuffle(x, 2)

        x = torch.tanh(self.conv2(x)) / 2
        return x, [h1, h2, h3, h4]
    
    def create_zeros(self, dims, gpu, grad):
        tmp = (torch.zeros(*dims, requires_grad=grad), 
               torch.zeros(*dims, requires_grad=grad))
        tmp = (tmp[0].cuda(), tmp[1].cuda()) if gpu else tmp
        return tmp
        
    def create_hidden(self, batch_size, gpu=True, grad=True):
        h1 = self.create_zeros([batch_size, 512, 2, 2], gpu, grad)
        h2 = self.create_zeros([batch_size, 512, 4, 4], gpu, grad)
        h3 = self.create_zeros([batch_size, 256, 8, 8], gpu, grad)
        h4 = self.create_zeros([batch_size, 128, 16, 16], gpu, grad)
        return [h1, h2, h3, h4]

