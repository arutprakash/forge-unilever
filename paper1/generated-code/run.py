"""Train and evaluate the paper-derived models; run `python run.py --help`."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_dataset
from models import MODELS


def evaluate(model, loader, save_predictions=False):
    model.eval()
    squared = absolute = count = 0
    predictions, targets = [], []
    with torch.no_grad():
        for history, target in loader:
            prediction = model(history)
            error = prediction.double() - target.double()
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            count += error.numel()
            if save_predictions:
                predictions.append(prediction.numpy())
                targets.append(target.numpy())
    metrics = {'mse': squared / count, 'mae': absolute / count, 'elements': count}
    if save_predictions:
        return metrics, np.concatenate(predictions), np.concatenate(targets)
    return metrics


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    splits, dataset_info = load_dataset(args.dataset, args.lookback, args.horizon)
    configuration = vars(args).copy()
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in configuration.items()}
    provenance = {'configuration': configuration, 'dataset': dataset_info,
                  'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                  'paper_sha256': hashlib.sha256((Path(__file__).resolve().parent.parent / 'paper1.pdf').read_bytes()).hexdigest(),
                  'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'numpy': np.__version__, 'device': 'cpu', 'threads': 1},
                  'source_basis': 'Local paper1.pdf, Section 4, Section 5, Appendix B.2 and D.1; no author-code imports',
                  'results': []}
    (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    validation = DataLoader(splits['validation'], batch_size=args.batch_size, shuffle=False)
    testing = DataLoader(splits['test'], batch_size=args.batch_size, shuffle=False)
    for name in args.models:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        training = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True,
                              drop_last=False, generator=generator)
        model = MODELS[name](args.lookback, args.horizon)
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        history_rows = []
        best_loss, best_epoch, stale = float('inf'), 0, 0
        start = time.monotonic()
        if name != 'Repeat':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            for epoch in range(1, args.epochs + 1):
                model.train()
                total, count = 0.0, 0
                for history, target in training:
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(history) - target).square().mean()
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * target.numel()
                    count += target.numel()
                val_loss = evaluate(model, validation)['mse']
                history_rows.append({'epoch': epoch, 'train_mse': total / count,
                                     'validation_mse': val_loss})
                print(f'{name} epoch {epoch}: train={total/count:.6f} validation={val_loss:.6f}', flush=True)
                if val_loss < best_loss - args.min_delta:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    torch.save({'state_dict': model.state_dict(), 'model': name,
                                'configuration': configuration, 'dataset': dataset_info},
                               folder / 'best.pt')
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        metrics, predictions, targets = evaluate(model, testing, save_predictions=True)
        np.save(folder / 'predictions.npy', predictions)
        # Every model uses identical targets; keep one shared copy.
        np.save(args.output / 'targets.npy', targets)
        starts = np.array(splits['test'].starts, dtype=np.int64)
        np.save(args.output / 'test_target_start_rows.npy', starts)
        (folder / 'training.json').write_text(json.dumps(history_rows, indent=2) + '\n')
        row = {'model': name, **metrics, 'parameters': sum(p.numel() for p in model.parameters()),
               'best_epoch': best_epoch, 'epochs_run': len(history_rows),
               'validation_mse': best_loss if name != 'Repeat' else None,
               'test_windows': len(splits['test']), 'seconds': time.monotonic() - start}
        provenance['results'].append(row)
        (folder / 'metrics.json').write_text(json.dumps(row, indent=2) + '\n')
        (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'FINAL {name}: MSE={metrics["mse"]:.6f} MAE={metrics["mae"]:.6f}', flush=True)
    with (args.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance['results'][0]))
        writer.writeheader()
        writer.writerows(provenance['results'])


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=root / 'datasets/ETTh1.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results/etth1-336-96')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--lookback', type=int, default=336)
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-6)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
