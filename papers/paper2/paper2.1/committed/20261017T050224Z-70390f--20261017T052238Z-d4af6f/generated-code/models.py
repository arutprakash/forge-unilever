import torch
from torch import nn

class FITS(nn.Module):
    def __init__(self, lookback, horizon, cutoff=26):
        super().__init__()
        self.lookback, self.horizon, self.cutoff = lookback, horizon, cutoff
        self.length = lookback + horizon
        self.out_bins = int(cutoff * self.length / lookback)
        self.linear = nn.Linear(cutoff, self.out_bins, dtype=torch.complex64)
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        scale = (x.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        z = ((x-mean)/scale).transpose(1,2)
        f = torch.fft.rfft(z, dim=-1)[..., 1:self.cutoff+1]
        f = self.linear(f)
        padded = torch.zeros((*f.shape[:-1], self.length//2+1), dtype=f.dtype, device=f.device)
        padded[...,1:self.out_bins+1] = f
        y = torch.fft.irfft(padded, n=self.length, dim=-1) * (self.length/self.lookback)
        return y.transpose(1,2)*scale+mean
