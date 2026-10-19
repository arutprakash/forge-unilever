#!/usr/bin/env python3
"""Docker experiment orchestrator. No LLM ever receives this host process's access.

Usage: python3 forge.py run --paper FILE.pdf --dataset FILE.csv --author-repo URL
Only Python's standard library and Docker are needed on the host.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
import traceback

from audit import Journal, sha256, seal_run, verify_run
from configuration import load_config, from_inputs, validate_protocol, validate_dataset
from benchmark import score, usage

BASE = Path(__file__).resolve().parent
IMAGES = {name: f'forge-{name}:1' for name in ('runtime', 'controller', 'proxy')}


def digest(path):
    return sha256(path)


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def auth_path():
    return Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'auth.json'


def preflight():
    docker = shutil.which('docker')
    available = False
    if docker:
        try:
            available = subprocess.run([docker, 'info'], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=15).returncode == 0
        except subprocess.SubprocessError:
            available = False
    non_root = os.getuid() != 0
    return {'docker_cli': bool(docker), 'docker_daemon': available, 'host_user_non_root': non_root,
            'codex_login_cache_available': auth_path().is_file(),
            'ready': available and non_root and auth_path().is_file()}


def mount(source, target, read_only=False, volume=False):
    # --mount uses commas as separators; reject ambiguous paths rather than
    # accidentally broadening the mounted filesystem.
    source = str(source)
    if ',' in source or '\n' in source:
        raise ValueError('Mount paths cannot contain commas or newlines')
    kind = 'volume' if volume else 'bind'
    return ['--mount', f'type={kind},source={source},target={target}' + (',readonly' if read_only else '')]


def hardened(name, network='none', user=None):
    return ['docker', 'run', '--name', name, '--network', network,
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--pids-limit', '256', '--memory', '4g', '--cpus', '2',
            '--user', user or f'{os.getuid()}:{os.getgid()}',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g,mode=1777']


def worker_command(name, workspace, inputs, ipc, author_repo=None):
    command = hardened(name)
    command += mount(workspace, '/workspace') + mount(inputs, '/inputs', True)
    command += mount(ipc, '/ipc', volume=True)
    if author_repo is not None:
        command += mount(author_repo, '/author-repo', True)
    return command + ['-d', IMAGES['runtime'], 'python', '/opt/pipeline/worker.py']


def controller_command(name, network, ipc, proxy, login_volume, model=None, controller_audit=None):
    command = hardened(name, network)
    command += mount(ipc, '/ipc', True, volume=True)
    command += mount(login_volume, '/login', volume=True)
    if controller_audit is not None:
        command += mount(controller_audit, '/audit')
    for env_name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy'):
        command += ['-e', f'{env_name}=http://{proxy}:3128']
    command += ['-i', IMAGES['controller']]
    return command + ([model] if model else [])


def prompt(stage):
    if stage == 'design':
        return (BASE / 'prompts/design.md').read_text()
    suffix = 'author.md' if stage == 'author' else 'paper-only.md'
    return (BASE / 'prompts/common.md').read_text() + '\n' + (BASE / 'prompts' / suffix).read_text()


class Docker:
    def __init__(self, audit):
        self.audit = Path(audit)
        self.audit.mkdir(parents=True, exist_ok=True)
        self.commands = []
        self.journal = Journal(self.audit)

    def run(self, command, input_text=None, log=None, timeout=7200, check=True):
        self.commands.append(command)
        save(self.audit / 'docker-commands.json', self.commands)
        number = len(self.commands)
        folder = self.audit / 'commands'
        folder.mkdir(exist_ok=True)
        stdout_path = Path(log) if log else folder / f'{number:05d}.stdout.log'
        stderr_path = folder / f'{number:05d}.stderr.log'
        started = time.monotonic()
        self.journal.emit('command_started', command_id=number, argv=command,
                          input_sha256=hashlib.sha256(input_text.encode()).hexdigest() if input_text else None,
                          timeout_seconds=timeout, stdout=str(stdout_path), stderr=str(stderr_path))
        try:
            if log:
                with stdout_path.open('w') as stream, stderr_path.open('w') as errors:
                    result = subprocess.run(command, input=input_text, text=True, stdout=stream,
                                            stderr=errors, timeout=timeout)
            else:
                result = subprocess.run(command, input=input_text, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
                stdout_path.write_text(result.stdout or '')
                stderr_path.write_text(result.stderr or '')
        except BaseException as exc:
            self.journal.emit('command_error', command_id=number, error=type(exc).__name__,
                              message=str(exc), elapsed_seconds=time.monotonic()-started)
            raise
        self.journal.emit('command_finished', command_id=number, exit_code=result.returncode,
                          elapsed_seconds=time.monotonic()-started,
                          stdout_sha256=digest(stdout_path), stderr_sha256=digest(stderr_path))
        if check and result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}); see {stdout_path} and {stderr_path}.')
        return result

    def inspect(self, name, filename):
        value = json.loads(self.run(['docker', 'inspect', name]).stdout)[0]
        save(self.audit / filename, value)
        return value


def assert_worker_policy(info, stage):
    host = info['HostConfig']
    if host['NetworkMode'] != 'none' or host.get('Privileged') or not host['ReadonlyRootfs']:
        raise RuntimeError('Worker Docker isolation policy failed')
    if 'ALL' not in [x.upper() for x in host.get('CapDrop', [])]:
        raise RuntimeError('Worker must drop all capabilities')
    if info.get('Config', {}).get('User', '').split(':')[0] in ('', '0', 'root'):
        raise RuntimeError('Worker must run as a non-root UID')
    if not any(option.startswith('no-new-privileges') for option in host.get('SecurityOpt', [])):
        raise RuntimeError('Worker must enforce no-new-privileges')
    expected = {'/workspace': True, '/inputs': False, '/ipc': True}
    if stage == 'author':
        expected['/author-repo'] = False
    actual = {m['Destination']: m['RW'] for m in info['Mounts'] if m['Type'] in ('bind', 'volume')}
    if actual != expected:
        raise RuntimeError(f'Unexpected worker mounts: {actual}')


def file_inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or '.git' in path.parts:
            continue
        if path.is_file() and not path.is_symlink():
            result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    return result


def run_stage(stage, directory, inputs, repo, model, timeout, prefix, login_volume):
    directory.mkdir()
    workspace = directory / ('generated-code' if stage == 'paper-only' else 'workspace')
    workspace.mkdir()
    results = directory / 'results'
    results.mkdir()
    audit = directory / 'audit'
    audit.mkdir()
    docker = Docker(audit)
    network, ipc = f'{prefix}-net', f'{prefix}-ipc'
    worker, driver, proxy = (f'{prefix}-{name}' for name in ('worker', 'driver', 'proxy'))
    evaluator = f'{prefix}-evaluator'
    policy = {'stage': stage, 'fresh_session': True, 'worker_network': 'none',
              'author_repo_visible': stage == 'author', 'shared_host_history': False}
    controller_audit = audit / 'controller'
    controller_audit.mkdir()
    (audit / 'prompt.md').write_text(prompt(stage))
    started = time.monotonic()
    docker.journal.emit('stage_started', stage=stage, prompt_sha256=digest(audit / 'prompt.md'))
    try:
        docker.run(['docker', 'network', 'create', '--internal', network])
        network_info = json.loads(docker.run(['docker', 'network', 'inspect', network]).stdout)[0]
        save(audit / 'network-inspect.json', network_info)
        if not network_info.get('Internal'):
            raise RuntimeError('Controller network must be Docker-internal')
        docker.run(['docker', 'volume', 'create', ipc])
        docker.run(hardened(f'{prefix}-init', user='0:0') + ['--rm'] + mount(ipc, '/ipc', volume=True)
                   + [IMAGES['runtime'], 'python', '-c', 'import os; os.chmod("/ipc", 0o777)'])
        command = hardened(proxy, network) + ['-d', IMAGES['proxy']]
        docker.run(command)
        # Only proxy is dual-homed. Neither offline worker nor Codex controller
        # has a public network route. Proxy forwards approved CONNECTs only.
        docker.run(['docker', 'network', 'connect', 'bridge', proxy])
        docker.run(worker_command(worker, workspace, inputs, ipc, repo if stage == 'author' else None))
        info = docker.inspect(worker, 'worker-inspect.json')
        assert_worker_policy(info, stage)
        deadline = time.monotonic() + 30
        while True:
            ready = docker.run(['docker', 'exec', worker, 'test', '-S', '/ipc/worker.sock'], check=False)
            if ready.returncode == 0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Offline worker failed to start')
            time.sleep(1)
        # Docker Desktop may expose dormant kernel tunnel devices even with
        # network=none. Require every non-loopback device down and no IPv4 routes.
        # Probe actual namespace: no active external interface, no login or
        # sibling sources, no mounted Docker socket. No model has run yet.
        probe = ('import os,socket; '
                 'assert all(name=="lo" or not (int(open("/sys/class/net/"+name+"/flags").read(),16)&1) for _,name in socket.if_nameindex()), "active external interface"; '
                 'assert not open("/proc/net/route").read().splitlines()[1:], "IPv4 route present"; '
                 'assert not os.path.exists("/run/login.json"); '
                 'assert not os.path.exists("/login"); '
                 'assert not os.path.exists("/var/run/docker.sock"); '
                 + ('assert not os.path.exists("/author-repo"); ' if stage != 'author' else '')
                 + 'print("namespace and mount probes passed")')
        docker.run(['docker', 'exec', worker, 'python', '-c', probe], log=audit / 'isolation-probe.log')
        save(audit / 'policy.json', policy)
        print(f'{stage}: fresh Codex session started; transcript {audit / "codex.jsonl"}', flush=True)
        docker.run(controller_command(driver, network, ipc, proxy, login_volume, model, controller_audit), input_text=prompt(stage),
                   log=audit / 'codex.jsonl', timeout=timeout)
        driver_info = docker.inspect(driver, 'controller-inspect.json')
        driver_mounts = {m['Destination'] for m in driver_info['Mounts'] if m['Type'] in ('bind', 'volume')}
        if driver_mounts != {'/ipc', '/login', '/audit'}:
            raise RuntimeError('Controller mount policy failed')
        if driver_info['HostConfig']['NetworkMode'] != network:
            raise RuntimeError('Controller is not on the internal network')
        # Stop untrusted processes before a fresh evaluator reads artifacts.
        docker.run(['docker', 'stop', '--time', '2', worker])
        if stage == 'design':
            path = safe_file(workspace, 'protocol.json')
            if path.stat().st_size > 65536:
                raise ValueError('Oversized protocol file')
            protocol = validate_protocol(json.loads(path.read_text()))
            validate_dataset(inputs / 'dataset.csv', protocol)
            save(audit / 'artifacts.json', file_inventory(workspace))
            docker.journal.emit('stage_finished', stage=stage, status='protocol_validated',
                                elapsed_seconds=time.monotonic()-started)
            return protocol
        safe_file(workspace, 'implementation.md')
        snapshot = directory / 'replay-source'
        replay_root = directory / 'replay'
        replay_output = replay_root / 'output'
        replay_output.mkdir(parents=True)
        prepare_replay(workspace, snapshot, json.loads((inputs / 'protocol.json').read_text()))
        replay_name = prefix + '-replay'
        try:
            command = hardened(replay_name) + mount(snapshot, '/workspace', True)
            command += mount(inputs, '/inputs', True) + mount(replay_output, '/replay-output')
            command += [IMAGES['runtime'], 'python', '/opt/pipeline/replay.py']
            docker.run(command, log=audit / 'replay.log')
            docker.inspect(replay_name, 'replay-inspect.json')
        finally:
            docker.run(['docker', 'rm', '-f', replay_name], check=False)
        command = hardened(evaluator) + mount(replay_root, '/workspace', True)
        command += mount(workspace, '/original', True)
        command += mount(inputs, '/inputs', True) + mount(results, '/results')
        command += [IMAGES['runtime'], 'python', '/opt/pipeline/evaluate.py', '--original', '/original']
        docker.run(command, log=audit / 'evaluation.log')
        docker.inspect(evaluator, 'evaluator-inspect.json')
        save(audit / 'artifacts.json', file_inventory(workspace))
        policy.update(status='evaluated', replay_verified=True, elapsed_seconds=time.monotonic()-started,
                      llm_usage=usage(audit / 'codex.jsonl'))
        save(audit / 'policy.json', policy)
        docker.journal.emit('stage_finished', **policy)
        return json.loads((results / 'metrics.json').read_text())
    except BaseException as exc:
        policy.update(status='failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        save(audit / 'policy.json', policy)
        docker.journal.emit('stage_failed', **policy)
        raise
    finally:
        docker.run(['docker', 'logs', proxy], log=audit / 'egress.log', check=False)
        docker.run(['docker', 'logs', worker], log=audit / 'worker.log', check=False)
        for name in (driver, evaluator, worker, proxy):
            docker.run(['docker', 'rm', '-f', name], check=False)
        docker.run(['docker', 'network', 'rm', network], check=False)
        docker.run(['docker', 'volume', 'rm', ipc], check=False)


def safe_file(root, relative):
    root = Path(root).resolve()
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or '..' in parts:
        raise ValueError('Unsafe source artifact path')
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Symlink source artifact rejected')
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f'Missing source artifact: {relative}')
    return path


def prepare_replay(workspace, snapshot, protocol):
    declaration = safe_file(workspace, 'replay.json')
    if declaration.stat().st_size > 65536:
        raise ValueError('Oversized replay declaration')
    value = json.loads(declaration.read_text())
    if set(value) != {'files', 'commands'} or not isinstance(value['files'], list) or not value['files']:
        raise ValueError('Replay must declare files and commands')
    if len(value['files']) > 2000:
        raise ValueError('Too many replay files')
    snapshot.mkdir()
    for relative in value['files']:
        source = safe_file(workspace, relative)
        if Path(relative).parts[0] in ('output', 'audit') or source.suffix not in ('.py', '.json', '.pt', '.pth', '.npy', '.npz', '.pkl', '.joblib'):
            raise ValueError('Only source/checkpoints/metadata can enter replay')
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if not any(Path(name).suffix == '.py' for name in value['files']):
        raise ValueError('Replay requires actual source files')
    commands = value['commands']
    if not isinstance(commands, list) or [c.get('model') for c in commands] != protocol['models']:
        raise ValueError('Replay model coverage/order differs from protocol')
    for entry in commands:
        argv = entry.get('argv')
        if set(entry) != {'model', 'argv'} or not isinstance(argv, list) or len(argv) < 2:
            raise ValueError('Invalid replay command')
        if not all(isinstance(x, str) and len(x) < 100000 for x in argv) or argv[0] not in ('python', 'python3'):
            raise ValueError('Replay commands must be Python argv arrays')
    save(snapshot / 'replay.json', value)
    return value


def configure_images(config):
    fingerprint = hashlib.sha256(json.dumps({'cli': config['codex_version'], 'sources': file_inventory(BASE / 'docker')}, sort_keys=True).encode()).hexdigest()[:12]
    IMAGES.update({name: f'forge-{name}:{fingerprint}' for name in IMAGES})


def build(config, docker=None):
    configure_images(config)
    docker = docker or Docker(BASE / 'builds' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    for name in IMAGES:
        command = ['docker', 'build', '--platform', 'linux/amd64', '--target', name,
                   '--build-arg', f'CODEX_VERSION={config["codex_version"]}', '-t', IMAGES[name], str(BASE / 'docker')]
        print(f'Building {name}; logs: {docker.audit}', flush=True)
        docker.run(command, log=docker.audit / f'build-{name}.log')


def write_report(run_root, config, author, generated, stages):
    report = score(author, generated, config['benchmark_policy'], stages, config.get('published_reference'))
    save(run_root / 'benchmark.json', report)
    # Keep a simple comparison artifact as well as the complete scoring record.
    save(run_root / 'comparison.json', {'models': report['models'], 'outcome': report['outcome']})
    lines = ['# ' + config['experiment'] + ' reproduction benchmark', '',
             f'Outcome: {report["outcome"]}. Execution rate: {report["execution_rate"]:.0%}.', '',
             'Reference: fresh author inference replay; published numbers are context only.', '',
             '| Model | Metric | Author | Generated | Relative gap | Within tolerance |',
             '|---|---|---:|---:|---:|---|']
    for row in report['models']:
        for metric, values in row['metrics'].items():
            gap = values['relative_gap']
            label = f'{gap:+.2%}' if gap is not None else 'undefined (zero reference)'
            lines.append(f'| {row["model"]} | {metric} | {values["author"]:.6f} | {values["generated"]:.6f} | {label} | {values["within_tolerance"]} |')
    lines += ['', '## Execution and provenance', '']
    for stage, status in stages.items():
        lines.append(f'- {stage}: {status["status"]}; {status.get("elapsed_seconds", 0):.1f} seconds. {status.get("error", "")}')
    lines += ['', 'Inspect stage implementation reports, replay execution records, and audit traces for method fidelity or copied-result concerns. Numerical agreement is not proof of architectural correctness.', '']
    (run_root / 'RESULTS.md').write_text('\n'.join(lines))
    return report


def run_experiment(config, model, run_id=None, build_first=False):
    configure_images(config)
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', run_id):
        raise ValueError('Unsafe run-id')
    run_root = Path(config.get('output_root', str(BASE / 'runs'))) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    audit = run_root / 'audit'
    docker = Docker(audit)
    manifest = {'experiment': config['experiment'], 'run_id': run_id, 'status': 'running',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'model_requested': model,
                'config': config, 'pipeline_sources': file_inventory(BASE / 'docker'),
                'orchestrator_sha256': digest(__file__), 'prompts': file_inventory(BASE / 'prompts')}
    manifest['host_sources'] = {p.name: digest(p) for p in sorted(BASE.glob('*.py'))}
    save(run_root / 'manifest.json', manifest)
    save(run_root / 'benchmark-policy.json', config['benchmark_policy'])
    docker.journal.emit('run_started', run_id=run_id, benchmark_policy_sha256=digest(run_root / 'benchmark-policy.json'))
    login_volume = None
    stages, outputs = {}, {'author': None, 'paper-only': None}
    started = time.monotonic()
    try:
        state = preflight()
        docker.journal.emit('preflight', **state)
        if not state['ready']:
            raise RuntimeError(f'Preflight failed: {state}. Start Docker and provide a file-backed Codex ChatGPT login.')
        login = json.loads(auth_path().read_text())
        if not login.get('tokens') or login.get('auth_mode') not in (None, 'chatgpt'):
            raise RuntimeError('ChatGPT Codex login required; API-key mode is not used')
        del login
        if build_first:
            build(config, docker)
        inputs = run_root / 'inputs'
        inputs.mkdir()
        shutil.copyfile(config['paper'], inputs / 'paper.pdf')
        shutil.copyfile(config['dataset'], inputs / 'dataset.csv')
        manifest['images'] = {}
        for name, tag in IMAGES.items():
            info = json.loads(docker.run(['docker', 'image', 'inspect', tag]).stdout)[0]
            manifest['images'][name] = {'tag': tag, 'id': info['Id'], 'repo_digests': info.get('RepoDigests', [])}
        docker.run(['docker', 'version'], log=audit / 'docker-version.log', timeout=30)
        docker.run(hardened('forge-environment-'+uuid.uuid4().hex[:10]) + ['--rm', IMAGES['runtime'],
                   'python', '-m', 'pip', 'freeze'], log=audit / 'runtime-packages.log', timeout=60)
        login_volume = 'forge-login-' + uuid.uuid4().hex[:12]
        docker.run(['docker', 'volume', 'create', login_volume])
        docker.run(hardened(login_volume+'-permissions', user='0:0') + ['--rm'] + mount(login_volume, '/login', volume=True)
                   + [IMAGES['runtime'], 'python', '-c', 'import os; os.chmod("/login",0o1777)'])
        bootstrap = ('from pathlib import Path; import shutil,os; Path("/login/cache").mkdir(mode=0o700); '
                     'shutil.copyfile("/run/login.json","/login/cache/auth.json"); os.chmod("/login/cache/auth.json",0o600)')
        docker.run(hardened(login_volume+'-bootstrap') + ['--rm'] + mount(auth_path().resolve(), '/run/login.json', True)
                   + mount(login_volume, '/login', volume=True) + [IMAGES['runtime'], 'python', '-c', bootstrap])
        prefix = 'forge-' + uuid.uuid4().hex[:12]
        if config.get('protocol') is None:
            docker.journal.emit('protocol_design_started')
            config['protocol'] = run_stage('design', run_root / 'protocol-design', inputs, None, model,
                                          config['stage_timeout_seconds'], prefix+'-design', login_volume)
            config['protocol_origin'] = 'validated fresh PDF-only design session'
        config['protocol'] = validate_protocol(config['protocol'])
        validate_dataset(inputs / 'dataset.csv', config['protocol'])
        save(inputs / 'protocol.json', config['protocol'])
        manifest['inputs'] = file_inventory(inputs)
        save(run_root / 'manifest.json', manifest)
        docker.journal.emit('protocol_frozen', protocol_sha256=digest(inputs / 'protocol.json'), origin=config.get('protocol_origin', 'configured'))
        author = run_root / 'author'
        author.mkdir()
        repo = author / 'repo'
        repo.mkdir()
        clone_name = prefix+'-fetch'
        fetch_script = ('import subprocess,sys; u,r=sys.argv[1:]; '
                        'subprocess.run(["git","config","--global","--add","safe.directory","/repo"],check=True); '
                        'subprocess.run(["git","init","/repo"],check=True,stdout=sys.stderr); '
                        'subprocess.run(["git","-C","/repo","remote","add","origin",u],check=True); '
                        'subprocess.run(["git","-C","/repo","fetch","--depth","1","origin",r],check=True,stdout=sys.stderr); '
                        'subprocess.run(["git","-C","/repo","checkout","--detach","FETCH_HEAD"],check=True,stdout=sys.stderr); '
                        'print(subprocess.check_output(["git","-C","/repo","rev-parse","HEAD"],text=True).strip())')
        fetched_ok = False
        try:
            command = hardened(clone_name, 'bridge') + mount(repo, '/repo')
            command += ['-e', 'GIT_TERMINAL_PROMPT=0', '-e', 'GIT_CONFIG_GLOBAL=/tmp/forge-gitconfig', IMAGES['runtime'], 'python', '-c', fetch_script,
                        config['author']['url'], config['author']['ref']]
            result = docker.run(command, timeout=600)
            manifest['author_commit'] = result.stdout.strip()
            if not re.fullmatch(r'[0-9a-f]{40}', manifest['author_commit']):
                raise ValueError('No valid author commit SHA')
            fetched_ok = True
            docker.journal.emit('author_fetched', commit=manifest['author_commit'])
        except Exception as exc:
            stages['author'] = {'status': 'fetch_error', 'error': str(exc)}
            docker.journal.emit('author_fetch_failed', error=str(exc))
        finally:
            docker.run(['docker', 'rm', '-f', clone_name], check=False)
        for stage, directory, repository, suffix in (
                ('author', author / 'run', repo, '-a'),
                ('paper-only', run_root / 'paper-only', None, '-g')):
            if stage == 'author' and not fetched_ok:
                continue
            phase_start = time.monotonic()
            docker.journal.emit('stage_dispatched', stage=stage)
            try:
                outputs[stage] = run_stage(stage, directory, inputs, repository, model,
                                           config['stage_timeout_seconds'], prefix+suffix, login_volume)
                stages[stage] = {'status': 'evaluated', 'elapsed_seconds': time.monotonic()-phase_start,
                                 'llm_usage': usage(directory / 'audit/codex.jsonl')}
            except Exception as exc:
                stages[stage] = {'status': 'execution_error', 'error': str(exc),
                                 'elapsed_seconds': time.monotonic()-phase_start}
                (audit / f'{stage}-error.log').write_text(traceback.format_exc())
            docker.journal.emit('stage_returned', stage=stage, **stages[stage])
        report = write_report(run_root, config, outputs['author'], outputs['paper-only'], stages)
        manifest['benchmark_outcome'] = report['outcome']
        manifest['stages'] = stages
        manifest['status'] = 'complete' if all(outputs.values()) else 'partial' if any(outputs.values()) else 'failed'
        if manifest['status'] != 'complete':
            raise RuntimeError(f'Incomplete execution; inspect {run_root / "RESULTS.md"}')
        print(f'Completed: {run_root / "RESULTS.md"}', flush=True)
    except BaseException as exc:
        if manifest['status'] == 'running':
            manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        (audit / 'failure.log').write_text(traceback.format_exc())
        docker.journal.emit('run_failed', error=str(exc), status=manifest['status'])
        if not (run_root / 'benchmark.json').exists():
            for stage in ('author', 'paper-only'):
                stages.setdefault(stage, {'status': 'not_started', 'error': 'Earlier infrastructure or protocol-design failure'})
            write_report(run_root, config, outputs['author'], outputs['paper-only'], stages)
            manifest['benchmark_outcome'] = 'execution_failure'
        raise
    finally:
        if login_volume is not None:
            docker.run(['docker', 'volume', 'rm', login_volume], check=False)
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['elapsed_seconds'] = time.monotonic()-started
        docker.journal.emit('run_finished', status=manifest['status'], elapsed_seconds=manifest['elapsed_seconds'])
        manifest['event_chain_tip'] = docker.journal.previous
        save(run_root / 'manifest.json', manifest)
        seal = seal_run(run_root)
        print(f'Audit: {run_root} (seal SHA-256 {seal})', flush=True)
    return run_root


def plan(config):
    return {'experiment': config['experiment'], 'config': config,
            'protocol_design': config.get('protocol') is None,
            'stages': ['optional PDF-only protocol design', 'author fetch', 'author train + isolated inference replay',
                       'fresh paper-only code generation + train + isolated inference replay', 'independent evaluation + benchmark'],
            'worker_network': 'none', 'paper_only_author_mount': False,
            'controller_network': 'internal; OpenAI inference/auth proxy only', 'fresh_sessions': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='run', choices=['preflight', 'plan', 'build', 'run', 'verify', 'benchmark', 'export'])
    parser.add_argument('--paper', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--author-repo')
    parser.add_argument('--author-ref', default='HEAD')
    parser.add_argument('--protocol', type=Path, help='Optional neutral forecasting protocol; otherwise preset or PDF-only design')
    parser.add_argument('--config', type=Path, help='Optional legacy/full experiment JSON')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--experiment')
    parser.add_argument('--model', help='Codex model available to your existing ChatGPT login')
    parser.add_argument('--run-id')
    parser.add_argument('--build', action='store_true', help='Rebuild images even if already available')
    parser.add_argument('--run-dir', type=Path, help='Run to verify')
    parser.add_argument('--evaluation-dir', type=Path, help='Completed evaluation to include in an export')
    parser.add_argument('--no-export', action='store_true', help='Skip automatic Git-trackable export')
    parser.add_argument('--benchmark-suite', choices=['none', 'autoreproduce'], default='none',
                        help='Optional upstream AutoReproduce evaluation after execution')
    args = parser.parse_args()
    if args.command == 'preflight':
        state = preflight()
        print(json.dumps(state, indent=2))
        return 0 if state['ready'] else 1
    if args.command == 'verify':
        if not args.run_dir:
            parser.error('verify requires --run-dir')
        print(json.dumps(verify_run(args.run_dir), indent=2))
        return 0
    if args.command == 'export':
        if not args.run_dir:
            parser.error('export requires --run-dir')
        from exporter import export_run
        export_run(args.run_dir, args.evaluation_dir)
        return 0
    if args.command == 'benchmark':
        if not args.run_dir:
            parser.error('benchmark requires --run-dir')
        from suites import evaluate_run
        evaluation = evaluate_run(args.run_dir, args.model)
        if not args.no_export:
            from exporter import export_run
            export_run(args.run_dir, evaluation)
        return 0
    if args.config:
        if any((args.paper, args.dataset, args.author_repo, args.protocol)):
            parser.error('Use either --config or the three input flags')
        config = load_config(args.config.resolve())
        if args.output_root:
            config['output_root'] = str(args.output_root.expanduser().resolve())
    else:
        if not all((args.paper, args.dataset, args.author_repo)):
            parser.error('--paper, --dataset, and --author-repo are required')
        config = from_inputs(args.paper, args.dataset, args.author_repo, args.protocol,
                             args.output_root, args.experiment, args.author_ref)
    if args.command == 'plan':
        print(json.dumps(plan(config), indent=2))
    elif args.command == 'build':
        build(config)
    else:
        # A single command builds missing images, then performs the entire run.
        configure_images(config)
        missing = False
        if shutil.which('docker'):
            missing = any(subprocess.run(['docker', 'image', 'inspect', tag], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL).returncode != 0 for tag in IMAGES.values())
        completed = run_experiment(config, args.model, args.run_id, args.build or missing)
        evaluation = None
        if args.benchmark_suite == 'autoreproduce':
            from suites import evaluate_run
            try:
                evaluation = evaluate_run(completed, args.model)
            except BaseException:
                if not args.no_export:
                    from exporter import export_run
                    export_run(completed)
                raise
        if not args.no_export:
            from exporter import export_run
            export_run(completed, evaluation)
    return 0


def cli():
    try:
        return main()
    except (ValueError, RuntimeError, subprocess.SubprocessError, OSError, KeyError, TypeError) as exc:
        print(f'forge: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(cli())
