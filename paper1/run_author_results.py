"""Run the unmodified author entry point on the repository's ETTh1 dataset."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'paper1/author-code/LTSF-Linear'
OUTPUT = ROOT / 'paper1/results/author-etth1'
DATA = ROOT / 'datasets/ETTh1.csv'

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ('DLinear', 'NLinear', 'Linear'):
        folder = OUTPUT / model
        folder.mkdir(exist_ok=True)
        args = ['--is_training', '1', '--root_path', str(DATA.parent),
                '--data_path', DATA.name, '--model_id', 'ETTh1_336_96',
                '--model', model, '--data', 'ETTh1', '--features', 'M',
                '--seq_len', '336', '--pred_len', '96', '--enc_in', '7',
                '--des', 'Exp', '--itr', '1', '--batch_size', '32',
                '--learning_rate', '0.005', '--num_workers', '0']
        bootstrap = ('import runpy,sys,torch; torch.set_num_threads(1); '
                     f'sys.path.insert(0,{str(AUTHOR)!r}); '
                     f'sys.argv={[str(AUTHOR / "run_longExp.py")] + args!r}; '
                     f'runpy.run_path({str(AUTHOR / "run_longExp.py")!r},run_name="__main__")')
        env = dict(os.environ, MPLCONFIGDIR=str(OUTPUT / '.matplotlib'),
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
        print(f'Running {model}; log: {folder / "run.log"}', flush=True)
        with (folder / 'run.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-c', bootstrap], cwd=folder,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (folder / 'run.log').read_text()
        matches = re.findall(r'mse:([\d.eE+-]+), mae:([\d.eE+-]+)', log)
        mse, mae = map(float, matches[-1])
        row = dict(model=model, mse=mse, mae=mae)
        rows.append(row)
        print(row, flush=True)
    import numpy, torch, pandas, sklearn
    metadata = dict(dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                    author_source='https://github.com/cure-lab/LTSF-Linear',
                    author_python_sha256={str(p.relative_to(AUTHOR)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(AUTHOR.rglob('*.py'))},
                    python=sys.version, numpy=numpy.__version__, torch=torch.__version__,
                    pandas=pandas.__version__, sklearn=sklearn.__version__,
                    seed=2021, seq_len=336, pred_len=96, learning_rate=0.005,
                    batch_size=32, train_epochs=10, patience=3, num_workers=0,
                    device='cpu', test_drop_last=False, results=rows)
    (OUTPUT / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (OUTPUT / 'summary.csv').write_text('model,mse,mae\n' + ''.join(f'{r["model"]},{r["mse"]},{r["mae"]}\n' for r in rows))

if __name__ == '__main__':
    main()
