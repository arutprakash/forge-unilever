"""Execute inference entrypoints in a fresh offline container, without cached predictions."""
import json
from pathlib import Path
import subprocess
import time


def main():
    declaration = json.loads(Path('/workspace/replay.json').read_text())
    events = []
    for entry in declaration['commands']:
        start = time.monotonic()
        with Path('/replay-output', entry['model']+'.log').open('w') as stream:
            result = subprocess.run(entry['argv'], cwd='/workspace', stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=600)
        events.append({'model': entry['model'], 'argv': entry['argv'], 'exit_code': result.returncode,
                       'elapsed_seconds': time.monotonic()-start})
        Path('/replay-output/execution.json').write_text(json.dumps(events, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'Inference replay failed: {entry["model"]}')


if __name__ == '__main__':
    main()
