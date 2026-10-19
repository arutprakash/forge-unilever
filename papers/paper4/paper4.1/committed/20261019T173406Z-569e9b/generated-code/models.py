import math
import torch
from torch import nn

class TimeBase(nn.Module):
    def __init__(self, lookback, horizon, period=24, rank=6):
        super().__init__()
        self.lookback, self.horizon, self.period = lookback, horizon, period
        self.extract = nn.Linear(math.ceil(lookback / period), rank)
        self.forecast = nn.Linear(rank, math.ceil(horizon / period))

    def forward(self, x, regularize=False):
        # [B,T,C] -> [B,C,P,N]; both linear maps act on segment index.
        x = x.transpose(1, 2)
        missing = (-self.lookback) % self.period
        if missing:
            # Complete last segment with matching phases from preceding segment.
            phase = self.lookback % self.period
            x = torch.cat([x, x[..., self.lookback-phase-self.period+phase:self.lookback-phase-self.period+phase+missing]], -1)
        b, c, t = x.shape
        basis = self.extract(x.reshape(b,c,-1,self.period).transpose(-1,-2))
        pred = self.forecast(basis).transpose(-1,-2).reshape(b,c,-1)[...,:self.horizon].transpose(1,2)
        if not regularize:
            return pred
        # Printed Eq. 5: X_basis^T X_basis where X_basis is [R,P].
        gram = basis @ basis.transpose(-1,-2)
        off = gram - torch.diag_embed(torch.diagonal(gram,dim1=-2,dim2=-1))
        orth = off.square().sum((-1,-2)).mean()
        return pred, orth
