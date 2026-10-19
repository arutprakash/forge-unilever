"""Forge's transparent scoring policy, inspired by the supplied workbook.

Not the official implementation of AutoReproduce, RECLAIM, or Artisan.
"""
import json
import math
from pathlib import Path


def usage(transcript):
    turns, counters, models = 0, {}, set()
    path = Path(transcript)
    if not path.exists():
        return {'turns': None, 'tokens': None, 'models': [], 'usd': None}
    with path.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('model'):
                models.add(event['model'])
            if event.get('type') == 'turn.completed':
                turns += 1
                for key, value in (event.get('usage') or {}).items():
                    if isinstance(value, (int, float)):
                        counters[key] = counters.get(key, 0) + value
    return {'turns': turns, 'tokens': counters or None, 'models': sorted(models), 'usd': None,
            'usd_note': 'Subscription cost is not inferred from token counts.'}


def relative_gap(candidate, reference):
    if reference == 0:
        return 0.0 if candidate == 0 else None
    return (candidate - reference) / abs(reference)


def score(author, generated, policy, stages, reference=None):
    tolerance = policy['relative_tolerance']
    if not 0 <= tolerance <= 1:
        raise ValueError('relative_tolerance must be between 0 and 1')
    report = {'policy': policy, 'published_reference': reference,
              'published_reference_used_for_pass_fail': False,
              'execution_rate': sum(s.get('status') == 'evaluated' for s in stages.values()) / 2,
              'stages': stages, 'models': [], 'method_fidelity': 'not automatically established',
              'copying_assessment': 'not automatically established; inspect source and action traces'}
    if author is None or generated is None:
        report.update(outcome='execution_failure', all_metrics_within_tolerance=False)
        return report
    if policy['require_prediction_replay'] and not (author.get('replay_verified') and generated.get('replay_verified')):
        raise ValueError('Benchmark requires independent inference replay for both stages')
    for field in ('protocol_sha256', 'dataset_sha256', 'prediction_shape'):
        if author[field] != generated[field]:
            raise ValueError(f'Comparison refused: different {field}')
    a = {row['model']: row for row in author['results']}
    g = {row['model']: row for row in generated['results']}
    if a.keys() != g.keys():
        raise ValueError('Comparison refused: different model coverage')
    for model, original in a.items():
        candidate = g[model]
        metrics = {}
        for metric in policy['metrics']:
            av, gv = original[metric], candidate[metric]
            if not all(math.isfinite(x) and x >= 0 for x in (av, gv)):
                raise ValueError('Invalid comparison metric')
            gap = relative_gap(gv, av)
            metrics[metric] = {'author': av, 'generated': gv, 'difference': gv-av,
                               'relative_gap': gap, 'absolute_relative_gap': abs(gap) if gap is not None else None,
                               'within_tolerance': gap is not None and abs(gap) <= tolerance + 1e-12}
        report['models'].append({'model': model, 'metrics': metrics})
    matched = all(metric['within_tolerance'] for row in report['models'] for metric in row['metrics'].values())
    report.update(outcome='numerical_match' if matched else 'metric_mismatch', all_metrics_within_tolerance=matched)
    return report
