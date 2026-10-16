"""Independent chronological windowing; no external model repository imports."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForecastWindows(Dataset):
    def __init__(self, values, target_start, target_stop, lookback, horizon):
        self.values = values
        self.lookback = lookback
        self.horizon = horizon
        self.starts = range(max(target_start, lookback), target_stop - horizon + 1)
        if not self.starts:
            raise ValueError('Split is too short for requested lookback/horizon')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        t = self.starts[index]
        return (torch.from_numpy(self.values[t-self.lookback:t]),
                torch.from_numpy(self.values[t:t+self.horizon]))


def load_dataset(path, lookback, horizon):
    frame = pd.read_csv(Path(path))
    dates = pd.to_datetime(frame['date'], errors='raise')
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError('Expected strictly increasing timestamps')
    if not (dates.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError('ETTh1 must have consecutive hourly observations')
    columns = [name for name in frame.columns if name != 'date']
    raw = frame[columns].to_numpy(dtype=np.float64)
    if raw.shape[1] != 7 or not np.isfinite(raw).all():
        raise ValueError('Expected seven finite-valued feature columns')
    n = len(raw)
    train_stop, val_stop = int(n * 0.6), int(n * 0.8)
    mean = raw[:train_stop].mean(axis=0)
    scale = raw[:train_stop].std(axis=0)
    scale[scale == 0] = 1.0
    values = ((raw - mean) / scale).astype(np.float32)
    boundaries = {'train': (0, train_stop), 'validation': (train_stop, val_stop),
                  'test': (val_stop, n)}
    splits = {name: ForecastWindows(values, start, stop, lookback, horizon)
              for name, (start, stop) in boundaries.items()}
    metadata = {'rows': n, 'columns': columns, 'mean': mean.tolist(),
                'scale': scale.tolist(), 'split_fractions': [0.6, 0.2, 0.2],
                'splits': {name: {'target_rows_start': start, 'target_rows_stop_exclusive': stop,
                                  'target_time_start': str(dates.iloc[start]),
                                  'target_time_end': str(dates.iloc[stop-1]),
                                  'windows': len(splits[name])}
                           for name, (start, stop) in boundaries.items()}}
    return splits, metadata
