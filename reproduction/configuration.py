"""Validate neutral forecasting protocols and resolve the three user inputs."""
import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SAFE = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,79}')


def validate_protocol(protocol):
    # Do not pass arbitrary agent-produced metadata, URLs, code, or instructions
    # to subsequent sessions. Only these typed experimental controls survive.
    allowed = {'dataset', 'models', 'lookback', 'horizon', 'features', 'split',
               'standardization', 'seed', 'training', 'evaluation', 'timestamp_column', 'interval_seconds'}
    if set(protocol) - allowed:
        raise ValueError('Unknown fields in neutral protocol')
    p = copy.deepcopy(protocol)
    for key in ('models', 'features'):
        if not isinstance(p[key], list) or not p[key] or len(set(p[key])) != len(p[key]):
            raise ValueError(f'{key} must be a nonempty list of unique names')
        valid = (lambda x: isinstance(x, str) and bool(SAFE.fullmatch(x))) if key == 'models' else (lambda x: isinstance(x, str) and 0 < len(x) <= 120 and not any(c in x for c in '\r\n\x00'))
        if not all(valid(x) for x in p[key]):
            raise ValueError(f'Unsafe {key} name')
    if len(p['models']) > 20 or len(p['features']) > 1000:
        raise ValueError('Protocol exceeds supported size')
    if not isinstance(p['dataset'], str) or not SAFE.fullmatch(p['dataset']):
        raise ValueError('Unsafe dataset label')
    for key in ('lookback', 'horizon', 'seed'):
        if type(p[key]) is not int or p[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid {key}')
    p.setdefault('timestamp_column', 'date')
    p.setdefault('interval_seconds', 3600)
    if not SAFE.fullmatch(p['timestamp_column']) or type(p['interval_seconds']) is not int or p['interval_seconds'] <= 0:
        raise ValueError('Invalid timestamp settings')
    split = p['split']
    if set(split) != {'train_end', 'validation_end', 'test_end'} or not all(type(x) is int for x in split.values()):
        raise ValueError('Invalid split schema')
    a, b, c = (split[k] for k in ('train_end', 'validation_end', 'test_end'))
    if not 0 < a < b < c or a < p['lookback'] + p['horizon'] or min(b-a, c-b) < p['horizon']:
        raise ValueError('Insufficient or unordered chronological splits')
    if p['standardization'] != 'per-channel training mean and population standard deviation':
        raise ValueError('Unsupported scaling rule')
    training = p['training']
    expected_training = {'optimizer', 'learning_rate', 'batch_size', 'max_epochs', 'patience',
                         'schedule', 'checkpoint_selection', 'training_shuffle', 'drop_last'}
    if set(training) != expected_training:
        raise ValueError('Invalid training schema')
    if training['optimizer'] not in ('Adam', 'AdamW', 'SGD', 'OLS', 'Ridge'):
        raise ValueError('Unsupported optimizer')
    for key in ('batch_size', 'max_epochs', 'patience'):
        if type(training[key]) is not int or not 1 <= training[key] <= 10000:
            raise ValueError(f'Invalid training {key}')
    lr = training['learning_rate']
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning_rate must be positive (ignored for a closed-form solver)')
    if training['schedule'] != 'constant' or training['checkpoint_selection'] != 'lowest validation MSE':
        raise ValueError('Unsupported schedule or checkpoint selection')
    if type(training['training_shuffle']) is not bool or training['drop_last'] is not False:
        raise ValueError('Training must include partial batches')
    expected_evaluation = {'units': 'standardized', 'metrics': ['mse', 'mae'], 'stride': 1,
                           'all_test_windows': True, 'test_only_after_selection': True}
    if p['evaluation'] != expected_evaluation:
        raise ValueError('Evaluator requires standardized MSE/MAE, stride one, full coverage, held-out test')
    return p


def validate_dataset(path, protocol):
    previous, count = None, 0
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = [protocol['timestamp_column'], *protocol['features']]
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f'CSV must contain {required}')
        for row in reader:
            stamp = datetime.fromisoformat(row[protocol['timestamp_column']].replace('Z', '+00:00'))
            if previous is not None and (stamp - previous).total_seconds() != protocol['interval_seconds']:
                raise ValueError(f'Irregular or unordered timestamp at CSV data row {count}')
            if not all(math.isfinite(float(row[name])) for name in protocol['features']):
                raise ValueError(f'Non-finite feature at CSV data row {count}')
            previous, count = stamp, count+1
    if count < protocol['split']['test_end']:
        raise ValueError('Dataset is shorter than the benchmark split')
    return count


def validate_config(config):
    for key in ('paper', 'dataset'):
        path = Path(config[key]).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f'{key} must be a file')
        config[key] = str(path)
    if Path(config['paper']).suffix.lower() != '.pdf' or Path(config['dataset']).suffix.lower() != '.csv':
        raise ValueError('This adapter accepts a PDF and a forecasting CSV')
    if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?', config['author']['url']):
        raise ValueError('Only public HTTPS github.com repository URLs are accepted')
    ref = config['author']['ref']
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_./-]*', ref) or '..' in ref:
        raise ValueError('Invalid author ref')
    if not re.fullmatch(r'\d+\.\d+\.\d+', config['codex_version']):
        raise ValueError('Pin a numeric Codex CLI version')
    if config.get('protocol') is not None:
        config['protocol'] = validate_protocol(config['protocol'])
        validate_dataset(config['dataset'], config['protocol'])
    policy = config.get('benchmark_policy', json.loads((BASE / 'benchmark-policy.json').read_text()))
    if policy['metrics'] != ['mse', 'mae'] or not 0 <= policy['relative_tolerance'] <= 1 or policy['require_prediction_replay'] is not True:
        raise ValueError('Unsupported benchmark policy')
    config['benchmark_policy'] = policy
    return config


def load_config(path):
    config = json.loads(path.read_text())
    for key in ('paper', 'dataset'):
        config[key] = str((path.parent / config[key]).resolve())
    config.setdefault('output_root', str(path.parent / 'runs'))
    return validate_config(config)


def from_inputs(paper, dataset, author_url, protocol_path=None, output_root=None,
                experiment=None, author_ref='HEAD'):
    paper, dataset = Path(paper).expanduser().resolve(), Path(dataset).expanduser().resolve()
    author_url = author_url.rstrip('/')
    # Only a host-side match. No repository URL or published reference enters a
    # protocol or the paper-only session.
    presets = [json.loads(p.read_text()) for p in sorted((BASE / 'presets').glob('*.json'))]
    preset = next((p for p in presets if p['author_url'].lower().removesuffix('.git').rstrip('/') == author_url.lower().removesuffix('.git').rstrip('/')), None)
    protocol = json.loads(Path(protocol_path).read_text()) if protocol_path else (preset['protocol'] if preset else None)
    if experiment is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}', experiment):
        raise ValueError('Unsafe experiment name')
    # Keep outputs in the repository even when a supplied PDF is in Downloads.
    paper_home = paper.parent if paper.is_relative_to(BASE.parent) else BASE.parent / 'papers' / paper.stem
    output = Path(output_root).expanduser().resolve() if output_root else paper_home / (experiment or '') / 'runs'
    return validate_config({'experiment': experiment or paper.stem, 'paper': str(paper), 'dataset': str(dataset),
                            'author': {'url': author_url, 'ref': author_ref}, 'codex_version': '0.160.1',
                            'stage_timeout_seconds': 7200, 'protocol': protocol, 'output_root': str(output),
                            'preset': preset['name'] if preset and not protocol_path else None,
                            'published_reference': preset.get('published_reference') if preset and not protocol_path else None,
                            'protocol_origin': 'user file' if protocol_path else ('workbook preset' if preset else 'pending PDF-only design')})
