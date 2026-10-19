"""Trusted evaluator: never import agent code or deserialize model checkpoints.

Run in a separate offline container after the model's worker is stopped.
Only fixed-format NumPy prediction arrays are read from untrusted artifacts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def regular_file(root, relative):
    root = root.resolve()
    path = root
    for part in Path(relative).parts:
        if part in ('..', '/'):
            raise ValueError('Unsafe artifact path')
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Symlink artifact rejected: {relative}')
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f'Missing/unsafe artifact: {relative}')
    return path


def load_data(dataset, protocol):
    # csv + NumPy only: independent of either implementation's data loader.
    with Path(dataset).open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    names = protocol['features']
    values = np.array([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    timestamps = np.array([row[protocol.get('timestamp_column', 'date')] for row in rows], dtype='datetime64[s]')
    if not np.all(np.diff(timestamps) == np.timedelta64(protocol.get('interval_seconds', 3600), 's')):
        raise ValueError('Dataset is not strictly ordered at the configured interval')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite dataset')
    split = protocol['split']
    a, b, c = (split[key] for key in ('train_end', 'validation_end', 'test_end'))
    if not 0 < a < b < c <= len(values):
        raise ValueError('Invalid split boundaries')
    mean = values[:a].mean(axis=0)
    scale = values[:a].std(axis=0)
    scale[scale == 0] = 1
    if not (np.isfinite(mean).all() and np.isfinite(scale).all()):
        raise ValueError('Non-finite training scaler')
    return (values - mean) / scale, mean, scale


def evaluate(dataset, protocol, workspace, destination, original=None):
    destination.mkdir(parents=True, exist_ok=True)
    data, mean, scale = load_data(dataset, protocol)
    horizon = protocol['horizon']
    split = protocol['split']
    starts = np.arange(split['validation_end'], split['test_end'] - horizon + 1)
    shape = (len(starts), horizon, len(protocol['features']))
    output = {'protocol_sha256': hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
              'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
              'prediction_shape': list(shape), 'test_target_start_rows': [int(starts[0]), int(starts[-1])],
              'mean': mean.tolist(), 'scale': scale.tolist(), 'units': 'standardized', 'results': []}
    output['replay_verified'] = original is not None
    for model in protocol['models']:
        path = regular_file(workspace, f'output/{model}.npy')
        max_size = int(np.prod(shape)) * 8 + 65536
        if path.stat().st_size > max_size:
            raise ValueError(f'Oversized predictions for {model}')
        predictions = np.load(path, mmap_mode='r', allow_pickle=False)
        if predictions.dtype not in (np.dtype('float32'), np.dtype('float64')) or predictions.shape != shape:
            raise ValueError(f'{model}: expected float32/64 {shape}, got {predictions.dtype} {predictions.shape}')
        initial = None
        if original is not None:
            original_path = regular_file(original, f'output/{model}.npy')
            if original_path.stat().st_size > max_size:
                raise ValueError('Oversized original predictions')
            initial = np.load(original_path, mmap_mode='r', allow_pickle=False)
            if initial.shape != shape or initial.dtype not in (np.dtype('float32'), np.dtype('float64')):
                raise ValueError('Invalid original prediction shape/type')
        squared = absolute = 0.0
        for offset in range(0, len(starts), 64):
            indices = starts[offset:offset+64]
            truth = np.stack([data[t:t+horizon] for t in indices])
            predicted = np.asarray(predictions[offset:offset+64], dtype=np.float64)
            if not np.isfinite(predicted).all():
                raise ValueError(f'Non-finite predictions: {model}')
            if initial is not None and not np.allclose(predicted, initial[offset:offset+64], rtol=1e-5, atol=1e-6):
                raise ValueError(f'Cached predictions do not match fresh inference replay: {model}')
            error = predicted - truth
            squared += np.square(error).sum(dtype=np.float64)
            absolute += np.abs(error).sum(dtype=np.float64)
        n = int(np.prod(shape))
        if not np.isfinite(squared) or not np.isfinite(absolute):
            raise ValueError(f'Non-finite aggregate metrics: {model}')
        output['results'].append({'model': model, 'mse': float(squared / n),
                                  'mae': float(absolute / n), 'forecast_elements': n,
                                  'predictions_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination / 'metrics.json').write_text(json.dumps(output, indent=2) + '\n')
    with (destination / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output['results'][0]))
        writer.writeheader()
        writer.writerows(output['results'])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/inputs/dataset.csv'))
    parser.add_argument('--protocol', type=Path, default=Path('/inputs/protocol.json'))
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--destination', type=Path, default=Path('/results'))
    parser.add_argument('--original', type=Path)
    args = parser.parse_args()
    evaluate(args.dataset, json.loads(args.protocol.read_text()), args.workspace, args.destination, args.original)
