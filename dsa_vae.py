import numpy as np
import torch
import torch.nn as nn


class FEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, f_size):
        super(ZEncoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.f_size = f_size 
        self.f_hidden = (torch.randn(2, batch_size,hidden_size),
                torch.randn(2, batch_size,hidden_size))
        self.f_lstm = nn.LSTM(input_size, hidden_size,1, bidirectional=True)
        
        self.f_mean = nn.Linear(2*hidden_size, f_size)
        self.f_var = nn.Linear(2*hidden_size, f_size)

    def forward(self,x):
       f_lstm_out,self.f_hidden = self.f_lstm(x,self.f_hidden)
       h_f,g_f = torch.split(f_lstm_out, self.hidden_size,dim=2)
       h_f = h_f[x.size(1)].view(-1,self.hidden_size)
       g_f = g_f[x.size(1)].view(-1,self.hidden_size)
       inputs =  torch.cat((h_f,g_f),dim=1)
       mean = self.f_mean(inputs)
       var = torch.exp(self.f_var(inputs))
       f_dist = torch.distributions.Normal(mean,torch.sqrt(var))
       sample_f = f_dist.sample()
       return mean,var,sample_f

class ZEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, z_size):
        super(ZEncoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.z_size = z_size
        self.z_hidden = (torch.randn(2, batch_size,hidden_size),
                torch.randn(2, batch_size,hidden_size))
        self.z_lstm = nn.LSTM(input_size, hidden_size, 1, bidirectional=True)
        self.z_rnn = 
