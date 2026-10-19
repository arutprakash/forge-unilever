import torch
from torch import nn
from torch.nn import functional as F

class Forecast(nn.Module):
    def __init__(self, name, lookback, horizon):
        super().__init__()
        if name not in ('Linear','NLinear','DLinear'): raise ValueError(name)
        self.name=name
        self.linear=nn.Linear(lookback,horizon,bias=False)
        if name=='DLinear': self.trend=nn.Linear(lookback,horizon,bias=False)
    def forward(self,x):
        z=x.transpose(1,2)
        if self.name=='NLinear':
            last=z[:,:,-1:]
            y=self.linear(z-last)+last
        elif self.name=='DLinear':
            trend=F.avg_pool1d(F.pad(z,(12,12),mode='replicate'),25,stride=1)
            y=self.linear(z-trend)+self.trend(trend)
        else: y=self.linear(z)
        return y.transpose(1,2)
