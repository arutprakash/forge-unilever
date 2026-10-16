"""LTSF-Linear models derived from Section 4 and Appendix B.2 of paper1.pdf.

Input/output axes: batch, time, channel. Temporal weights are shared across
channels. Biases are omitted to match W x and the paper's TL/2TL counts.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TemporalMap(nn.Module):
    def __init__(self, lookback, horizon):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(horizon, lookback))
        nn.init.uniform_(self.weight, -lookback ** -0.5, lookback ** -0.5)

    def forward(self, history):
        return torch.einsum('btc,ht->bhc', history, self.weight)


class Linear(nn.Module):
    def __init__(self, lookback, horizon):
        super().__init__()
        self.temporal = TemporalMap(lookback, horizon)

    def forward(self, history):
        return self.temporal(history)


class NLinear(Linear):
    def forward(self, history):
        anchor = history[:, -1:, :]
        return self.temporal(history - anchor) + anchor


class DLinear(nn.Module):
    def __init__(self, lookback, horizon, kernel=25):
        super().__init__()
        if kernel <= 0 or kernel % 2 != 1:
            raise ValueError('Moving-average kernel must be a positive odd integer')
        self.kernel = kernel
        self.trend_map = TemporalMap(lookback, horizon)
        self.remainder_map = TemporalMap(lookback, horizon)

    def decompose(self, history):
        # The paper leaves edge handling unspecified. Replicate endpoints so
        # the centered average keeps the input length and preserves constants.
        channels_first = history.transpose(1, 2)
        padded = F.pad(channels_first, (self.kernel // 2,) * 2, mode='replicate')
        trend = F.avg_pool1d(padded, self.kernel, stride=1).transpose(1, 2)
        return trend, history - trend

    def forward(self, history):
        trend, remainder = self.decompose(history)
        return self.trend_map(trend) + self.remainder_map(remainder)


class Repeat(nn.Module):
    """Last-observation baseline described in Section 5.1."""
    def __init__(self, lookback, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, history):
        return history[:, -1:, :].expand(-1, self.horizon, -1)


MODELS = {'Linear': Linear, 'NLinear': NLinear, 'DLinear': DLinear, 'Repeat': Repeat}
