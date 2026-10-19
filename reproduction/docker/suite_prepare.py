"""Offline adapter: invoke pinned upstream functions, capturing their prompts.

Never imports or executes candidate/reference code. Upstream source lives in
/upstream, sealed experiment in /run, and adapter outputs in /suite.
"""
import importlib.util
import json
from pathlib import Path
import sys
import types


def bundle(root):
    root = Path(root)
    paths = sorted(root.rglob('*.py'))
    if not paths:
        raise ValueError(f'No Python source in {root}')
    texts = []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe source path')
        texts.append(f'\n# Source: {path.relative_to(root)}\n' + path.read_text())
    text = '\n'.join(texts)
    if len(text.encode()) > 2_000_000:
        raise ValueError('Source bundle exceeds adapter budget; refusing to truncate')
    return text


def load(name):
    spec = importlib.util.spec_from_file_location(name, '/upstream/evaluation/' + name + '.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    stage = sys.argv[1]
    out = Path('/suite')
    experiment = json.loads(Path('/run/manifest.json').read_text())
    p = json.loads(Path('/run/inputs/protocol.json').read_text())
    # Upstream functions use module-level defaults from this config.
    cfg = types.ModuleType('evaluation.exp_config')
    cfg.MODEL, cfg.TASK, cfg.INSTRUCTION = 'custom', 'forecasting', 'forecasting'
    pkg = types.ModuleType('evaluation'); pkg.__path__ = []
    sys.modules['evaluation'] = pkg
    sys.modules['evaluation.exp_config'] = cfg
    inference = types.ModuleType('inference')
    def capture(**kwargs):
        (out / f'{stage}.prompt.txt').write_text(kwargs['prompt'])
        (out / f'{stage}.request.json').write_text(json.dumps({k:v for k,v in kwargs.items() if k != 'prompt'}, indent=2))
        return 'PROMPT_CAPTURED_ONLY'
    inference.query_model = capture
    sys.modules['inference'] = inference
    for directory in ('papers', 'points', 'generated/custom', 'generated/custom_repo', 'reference/custom', 'scratch'):
        (out / directory).mkdir(parents=True, exist_ok=True)
    instruction = ('Implement the paper model for forecasting on the provided CSV using the following frozen Forge protocol. '
                   'Protocol choices override paper settings only where explicitly specified; assess other method details against the paper.\n'
                   + json.dumps(p, sort_keys=True))
    if stage == 'points':
        from pypdf import PdfReader
        text = '\n'.join(f'\nPAGE {i+1}\n' + (page.extract_text() or '') for i,page in enumerate(PdfReader('/run/inputs/paper.pdf').pages))
        (out / 'papers/custom.txt').write_text(text)
        load('summarize_points').summarize_points('custom', 'forecasting', instruction,
            str(out / 'papers'), str(out / 'scratch'), 'codex-login-default')
    elif stage == 'gap':
        report = json.loads(Path('/run/benchmark.json').read_text())
        calculate = load('cal_per_gap').calculate_perf_gap
        rows = [{'model':row['model'], 'metric':metric, 'gap':float(calculate(v['author'], v['generated']))}
                for row in report['models'] for metric,v in row['metrics'].items()]
        (out / 'execution.json').write_text(json.dumps({'execution_rate':report['execution_rate'],
            'metric_gaps':rows, 'mean_gap_across_model_metrics':sum(r['gap'] for r in rows)/len(rows),
            'formula':'abs(reference-candidate)/max(reference,candidate); zero/zero=0',
            'scope':'custom FITS/task metrics; not the 13-task official aggregate'}, indent=2))
        return
    else:
        # Preserve all recursive source text in one file because upstream only
        # reads top-level *.py. Text is evaluated, never imported.
        generated = bundle('/run/paper-only/generated-code')
        reference = bundle('/run/author/run/workspace')
        (out / 'generated/custom/source.py').write_text(generated)
        (out / 'generated/custom_repo/source.py').write_text(generated)
        (out / 'reference/custom/run_custom.py').write_text(reference)
        if stage == 'high':
            load('eval_high').eval_high_level('custom', instruction, str(out/'points'),
                str(out/'generated'), str(out/'scratch'), 'codex-login-default')
        elif stage == 'low':
            load('eval_low').eval_low_level('custom', str(out/'reference'), str(out/'generated'),
                str(out/'scratch'), 'codex-login-default')
        elif stage == 'mixed':
            load('eval_mixed').eval_mixed_level('custom', instruction, str(out/'points'),
                str(out/'reference'), str(out/'generated'), str(out/'scratch'), 'codex-login-default')
        else:
            raise ValueError('Unknown evaluation stage')
    if not (out / f'{stage}.prompt.txt').is_file():
        raise RuntimeError('Upstream failed to construct prompt')


if __name__ == '__main__':
    main()
