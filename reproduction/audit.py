"""Host-owned append-only event journal with a verifiable SHA-256 chain."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def redact(value):
    if isinstance(value, str):
        return re.sub(r'(?i)(?:sk-(?:proj-)?[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._-]{12,})', '[REDACTED]', value)
    if isinstance(value, dict):
        return {k: '[REDACTED]' if any(s in k.lower() for s in ('access_token', 'refresh_token', 'api_key', 'id_token')) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'events.jsonl'
        self.sequence, self.previous = 0, '0' * 64
        if self.path.exists():
            self.sequence, self.previous = verify_chain(self.path)

    def emit(self, event, **details):
        self.sequence += 1
        record = {'sequence': self.sequence, 'utc': datetime.now(timezone.utc).isoformat(),
                  'event': event, 'details': redact(details), 'previous_sha256': self.previous}
        record['sha256'] = hashlib.sha256(canonical(record)).hexdigest()
        with self.path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
        self.previous = record['sha256']
        return record


def verify_chain(path):
    previous, sequence = '0' * 64, 0
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            claimed = record.pop('sha256')
            sequence += 1
            if record['sequence'] != sequence or record['previous_sha256'] != previous:
                raise ValueError(f'Broken audit sequence: {path}:{sequence}')
            previous = hashlib.sha256(canonical(record)).hexdigest()
            if previous != claimed:
                raise ValueError(f'Altered audit record: {path}:{sequence}')
    return sequence, previous


def seal_run(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.name != 'audit-seal.json':
            files[str(path.relative_to(root))] = sha256(path)
    seal = {'files': files, 'note': 'Local hash seal; archive its hash externally to detect replacement of the entire seal.'}
    (root / 'audit-seal.json').write_text(json.dumps(seal, indent=2) + '\n')
    return sha256(root / 'audit-seal.json')


def verify_run(root):
    root = Path(root).resolve()
    seal = json.loads((root / 'audit-seal.json').read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.name != 'audit-seal.json'}
    if actual != set(seal['files']):
        raise ValueError('Run contains missing or unsealed extra files')
    for relative, expected in seal['files'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Missing or altered artifact: {relative}')
    for path in [root / 'audit/events.jsonl', *root.glob('*/audit/events.jsonl'), *root.glob('*/*/audit/events.jsonl')]:
        if path.is_file():
            verify_chain(path)
    return {'verified_files': len(seal['files']), 'seal_sha256': sha256(root / 'audit-seal.json')}
