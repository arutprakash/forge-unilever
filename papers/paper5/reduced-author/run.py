"""Prepare and run a smaller author model without modifying sealed archives."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'paper5.1/runs/20261019T173137Z-383443'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--smoke', action='store_true', help='Check both models with a training batch')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--memory', default='4g')
    parser.add_argument('--cpus', default='2')
    args = parser.parse_args()
    workspace = HERE / 'runs' / 'small-v1'
    if not workspace.exists():
        workspace.mkdir(parents=True)
        original = SOURCE / 'author/run/workspace'
        for name in ('models', 'layers', 'utils'):
            shutil.copytree(original / name, workspace / name, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('AUTHOR_LICENSE', 'run_all.py', 'experiment.py'):
            shutil.copy2(original / name, workspace / name)
        driver = workspace / 'experiment.py'
        text = driver.read_text().replace('d_model=512', 'd_model=64').replace('d_ff=2048', 'd_ff=256')
        # Keep optimizer batches intact; smaller wavelet channels permit full batches.
        text = text.replace("micro=1 if name=='FEDformer-w' else len(ids)", 'micro=len(ids)')
        driver.write_text(text)
        model = workspace / 'models/FEDformer.py'
        text = model.read_text().replace('MultiWaveletTransform(ich=', 'MultiWaveletTransform(c=8, ich=')
        text = text.replace('MultiWaveletCross(in_channels=', 'MultiWaveletCross(c=8, in_channels=')
        model.write_text(text)
        shutil.copy2(SOURCE / 'inputs/protocol.json', workspace / 'protocol.json')
        (workspace / 'architecture.json').write_text(json.dumps(dict(
            label='reduced-size author implementation; not original paper configuration',
            source_run=str(SOURCE), d_model=64, d_ff=256, n_heads=8,
            wavelet_self_channels=8, wavelet_cross_channels=8, wavelet_degree=8,
            protocol='unchanged from source run',
        ), indent=2) + '\n')
    if args.prepare_only:
        print(workspace)
        return
    image = json.loads((SOURCE / 'manifest.json').read_text())['images']['runtime']['id']
    command = ['python', '/runner/smoke.py'] if args.smoke else ['python', 'run_all.py']
    subprocess.run([
        'docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none', '--read-only',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit', '256',
        '--memory', args.memory, '--cpus', args.cpus, '--user', f'{os.getuid()}:{os.getgid()}',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g,mode=1777',
        '--mount', f'type=bind,source={workspace},target=/workspace',
        '--mount', f'type=bind,source={SOURCE / "inputs"},target=/inputs,readonly',
        '--mount', f'type=bind,source={HERE},target=/runner,readonly',
        image, *command,
    ], check=True)


if __name__ == '__main__':
    main()
