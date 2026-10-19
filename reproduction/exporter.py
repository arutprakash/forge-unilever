"""Small, immutable, Git-trackable exports of verified experiment artifacts."""
import json
from pathlib import Path
import shutil
import tempfile

from audit import sha256, seal_run, verify_run


def export_run(run_dir, evaluation_dir=None):
    run = Path(run_dir).resolve()
    verified = verify_run(run)
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Only complete, sealed runs can be exported')
    evaluation = Path(evaluation_dir).resolve() if evaluation_dir else None
    suite = None
    if evaluation:
        evaluation_seal = verify_run(evaluation)
        suite = json.loads((evaluation/'suite-results.json').read_text())
        if suite['status'] != 'complete' or Path(suite['source_run']).resolve() != run or suite['source_seal'] != verified:
            raise ValueError('Evaluation must be complete and match this exact sealed run')
    parent = run.parent.parent/'committed'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent/run.name
    # Each run is an immutable snapshot. Later benchmark results get their own
    # export rather than overwriting a previously committed numerical-only one.
    if evaluation:
        target = parent/(run.name+'--'+evaluation.name)
    if target.exists():
        verify_run(target)
        existing = json.loads((target/'provenance.json').read_text())
        if existing['source_seal'] != verified or existing.get('evaluation_seal') != (evaluation_seal if evaluation else None):
            raise ValueError('Existing export has different provenance')
        print(f'Committed-artifact export already exists: {target}', flush=True)
        return target
    with tempfile.TemporaryDirectory(prefix='.export-', dir=parent) as folder:
        root = Path(folder)
        copied = {}
        def copy(source, destination):
            source = Path(source)
            if source.is_symlink() or any(p.is_symlink() for p in source.parents):
                raise ValueError(f'Symlink export rejected: {source}')
            if not source.is_file() or source.stat().st_size > 2_000_000:
                raise ValueError(f'Missing or oversized text artifact: {source}')
            source.read_text(encoding='utf-8')
            dest = root/destination
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            copied[str(destination)] = {'source':str(source.relative_to(evaluation if evaluation and source.is_relative_to(evaluation) else run)),
                                        'origin':'evaluation' if evaluation and source.is_relative_to(evaluation) else 'run',
                                        'sha256':sha256(source)}
        generated = run/'paper-only/generated-code'
        # Never recursively collect arbitrary agent output or author sources.
        for path in sorted(generated.rglob('*')):
            relative = path.relative_to(generated)
            if any(p in ('output','checkpoints','__pycache__','.git','.venv') or p.startswith('replay') for p in relative.parts[:-1]):
                continue
            if path.is_file() and (path.suffix == '.py' or path.name in ('requirements.txt','pyproject.toml','README.md','implementation.md','replay.json')):
                copy(path, Path('generated-code')/relative)
        if not any(p.endswith('.py') for p in copied):
            raise ValueError('No generated Python source to export')
        copy(run/'inputs/protocol.json', Path('protocol.json'))
        for name in ('RESULTS.md','benchmark.json','comparison.json','benchmark-policy.json'):
            copy(run/name, Path('results')/name)
        for stage, subdir in (('author','author/run'),('generated','paper-only')):
            copy(run/subdir/'results/metrics.json', Path('results')/(stage+'-metrics.json'))
        if evaluation:
            for name in ('RESULTS.md','suite-results.json','execution.json','points.txt','high.txt','low.txt','mixed.txt'):
                copy(evaluation/name, Path('benchmark')/name)
        config = manifest['config']
        provenance = {'run_id':run.name,'experiment':manifest['experiment'],
                      'started_utc':manifest.get('started_utc'),'finished_utc':manifest.get('finished_utc'),
                      'stages':manifest.get('stages',{}),
                      'source_seal':verified,'evaluation_id':evaluation.name if evaluation else None,
                      'evaluation_seal':evaluation_seal if evaluation else None,
                      'author':config['author'],'author_commit':manifest['author_commit'],
                      'inputs':manifest['inputs'],'images':manifest['images'],
                      'pipeline_sources':manifest.get('host_sources',{}),
                      'copied_files':copied,
                      'exclusions':['author checkout','checkpoints','prediction arrays','dataset','paper PDF','raw transcripts','credentials'],
                      'note':'Curated subset; original full run/evaluation seals are references, not included full archives.'}
        (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        readme = f'''# {manifest['experiment']} — preserved experiment

Source run: `{run.name}`. Generated source is copied without modification.
See [numerical results](results/RESULTS.md), [protocol](protocol.json), and
[provenance](provenance.json). {'See [alignment evaluation](benchmark/RESULTS.md).' if evaluation else 'No alignment evaluation is included.'}

## Reproduce from the repository root

```bash
python3 forge.py run \\
  --paper PATH_TO_ORIGINAL_PDF \\
  --dataset PATH_TO_ORIGINAL_CSV \\
  --author-repo {config['author']['url']} \\
  --author-ref {manifest['author_commit']} \\
  --protocol {target.relative_to(Path(__file__).resolve().parents[1])/'protocol.json' if target.is_relative_to(Path(__file__).resolve().parents[1]) else 'PATH_TO_EXPORTED_PROTOCOL'} \\
  --experiment {manifest['experiment']}{' '+chr(92)+chr(10)+'  --benchmark-suite autoreproduce' if evaluation else ''}
```

Use the original input files matching the SHA-256 hashes in provenance. Runtime
dependencies and image identities are recorded there; the pipeline Dockerfile
provides the pinned environment. Code entry points are documented in
`generated-code/README.md`. Paths such as `/inputs` refer to container mounts.
Checkpoints are excluded, so inference replay requires retraining or the original
ignored run archive. Exact rerun scores may vary with model generation and training.

## Verify this export

```bash
python3 forge.py verify --run-dir {target.relative_to(Path(__file__).resolve().parents[1]) if target.is_relative_to(Path(__file__).resolve().parents[1]) else 'PATH_TO_EXPORT'}
```

Only small text source and reports are preserved. Original runs retain large
artifacts and raw audit traces. Benchmark alignment uses upstream evaluators on a
custom task and is not an official ReproduceBench leaderboard score. Generated
code may require license review before redistribution; no new license is assigned.
'''
        (root/'README.md').write_text(readme)
        seal_run(root)
        root.rename(target)
    print(f'Git-trackable export: {target}',flush=True)
    return target
