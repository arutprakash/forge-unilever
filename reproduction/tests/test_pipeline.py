"""Run with: python3 -m unittest discover -s tests -v (no Docker or login needed)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import pipeline


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / 'docker' / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


proxy = module('proxy', 'proxy.py')
bridge = module('bridge', 'bridge.py')


class IsolationContracts(unittest.TestCase):
    def test_paper_only_mounts_no_repo_history_or_login(self):
        command = pipeline.worker_command('worker', '/isolated/generated', '/clean/inputs', 'stage-ipc')
        mounts = [command[i+1] for i, arg in enumerate(command) if arg == '--mount']
        self.assertEqual(command[command.index('--network')+1], 'none')
        self.assertEqual(len(mounts), 3)
        self.assertTrue(any('source=/clean/inputs,target=/inputs,readonly' in value for value in mounts))
        text = ' '.join(mounts)
        for forbidden in ('author', '.codex', 'login', 'docker.sock', 'paper1.1'):
            self.assertNotIn(forbidden, text)
        author = pipeline.worker_command('w', '/workspace', '/inputs', 'ipc', '/author/source')
        self.assertIn('type=bind,source=/author/source,target=/author-repo,readonly', author)

    def test_controller_has_no_workspace_or_docker_socket(self):
        with patch.object(pipeline, 'auth_path', return_value=Path('/test/login.json')):
            command = pipeline.controller_command('driver', 'internal', 'ipc', 'proxy', 'login-state')
        mounts = [command[i+1] for i, arg in enumerate(command) if arg == '--mount']
        self.assertEqual(len(mounts), 2)
        self.assertIn('type=volume,source=ipc,target=/ipc,readonly', mounts)
        self.assertIn('type=volume,source=login-state,target=/login', mounts)
        self.assertTrue(all('type=volume' in item for item in mounts))
        self.assertFalse(any('workspace' in item or 'docker.sock' in item for item in mounts))
        self.assertNotIn('--privileged', command)

    def test_proxy_does_not_allow_github_web_registries_or_lookalikes(self):
        for host in ('github.com', 'raw.githubusercontent.com', 'pypi.org', 'google.com',
                     'chatgpt.com.evil.org', 'chatgpt.com@github.com', '127.0.0.1',
                     '169.254.169.254', 'host.docker.internal'):
            self.assertFalse(proxy.allowed_target('CONNECT', host + ':443'))
        self.assertFalse(proxy.allowed_target('GET', 'https://chatgpt.com/'))
        self.assertFalse(proxy.allowed_target('CONNECT', 'chatgpt.com:80'))
        self.assertTrue(proxy.allowed_target('CONNECT', 'chatgpt.com:443'))

    def test_inspection_rejects_extra_mount_and_network(self):
        info = {'HostConfig': {'NetworkMode': 'none', 'Privileged': False,
                               'ReadonlyRootfs': True, 'CapDrop': ['ALL'],
                               'SecurityOpt': ['no-new-privileges']},
                'Config': {'User': '501:20'},
                'Mounts': [{'Type': 'bind', 'Destination': '/workspace', 'RW': True},
                           {'Type': 'bind', 'Destination': '/inputs', 'RW': False},
                           {'Type': 'volume', 'Destination': '/ipc', 'RW': True}]}
        pipeline.assert_worker_policy(info, 'paper-only')
        info['Mounts'].append({'Type': 'bind', 'Destination': '/author-repo', 'RW': False})
        with self.assertRaises(RuntimeError):
            pipeline.assert_worker_policy(info, 'paper-only')
        info['Mounts'].pop()
        info['HostConfig']['NetworkMode'] = 'bridge'
        with self.assertRaises(RuntimeError):
            pipeline.assert_worker_policy(info, 'paper-only')

    def test_worker_cannot_use_root_uid(self):
        info = {'HostConfig': {'NetworkMode': 'none', 'Privileged': False,
                               'ReadonlyRootfs': True, 'CapDrop': ['ALL'],
                               'SecurityOpt': ['no-new-privileges']},
                'Config': {'User': '0:0'}, 'Mounts': []}
        with self.assertRaisesRegex(RuntimeError, 'non-root'):
            pipeline.assert_worker_policy(info, 'paper-only')

    def test_paper_prompt_contains_no_author_result_or_repo_url(self):
        prompt = pipeline.prompt('paper-only')
        self.assertNotIn('cure-lab', prompt)
        self.assertNotIn('https://github', prompt)
        self.assertNotIn('/author-repo', prompt)
        self.assertNotIn('0.3841', prompt)
        self.assertIn('solely from /inputs/paper.pdf', prompt)

    def test_mount_separator_injection_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.mount('/path,source=/secret', '/inputs')

    def test_bridge_exposes_only_offline_exec(self):
        listed = bridge.dispatch({'method': 'tools/list'})
        self.assertEqual([tool['name'] for tool in listed['tools']], ['exec'])
        with self.assertRaises(ValueError):
            bridge.dispatch({'method': 'tools/call', 'params': {'name': 'fetch_url'}})

    def test_mcp_stdio_handshake(self):
        messages = [
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}},
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}]
        result = subprocess.run([sys.executable, str(BASE / 'docker/bridge.py')],
                                input=''.join(json.dumps(m)+'\n' for m in messages),
                                text=True, capture_output=True, check=True)
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r['id'] for r in replies], [1, 2])
        self.assertEqual(replies[1]['result']['tools'][0]['name'], 'exec')


class Orchestration(unittest.TestCase):
    def setup_run(self, temporary):
        root = Path(temporary)
        for folder in ('docker', 'prompts'):
            (root / folder).mkdir()
        paper, dataset = root / 'paper.pdf', root / 'dataset.csv'
        paper.write_bytes(b'local-paper-only')
        # Actual shared dataset is copied so protocol validation remains active
        # even though Docker/model execution is mocked below.
        dataset.write_bytes((BASE.parent/'datasets/ETTh1.csv').read_bytes())
        login = root / 'login.json'
        login.write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {'access_token': 'test-only-not-a-real-token'}}))
        config = pipeline.load_config(BASE / 'experiment.json')
        config.update(paper=str(paper), dataset=str(dataset), output_root=str(root/'runs'))
        return root, login, config

    @staticmethod
    def fake_docker(self, command, **kwargs):
        if command[1:3] == ['image', 'inspect']:
            return SimpleNamespace(stdout=json.dumps([{'Id': 'sha256:test-image'}]), stderr='', returncode=0)
        if '-c' in command and 'git' in command[command.index('-c')+1]:
            return SimpleNamespace(stdout='a'*40+'\n', stderr='fetch complete\n', returncode=0)
        return SimpleNamespace(stdout='', stderr='', returncode=0)

    def test_comparison_only_after_two_separate_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, login, config = self.setup_run(temporary)
            calls = []

            def fake_stage(stage, directory, inputs, repo, model, timeout, prefix, login_volume):
                calls.append((stage, repo))
                self.assertEqual(sorted(p.name for p in inputs.iterdir()), ['dataset.csv', 'paper.pdf', 'protocol.json'])
                neutral = (inputs / 'protocol.json').read_text()
                self.assertNotIn('github.com', neutral)
                self.assertNotIn('author_mse', neutral)
                return {'protocol_sha256': 'same-protocol', 'dataset_sha256': 'same-data',
                        'prediction_shape': [2785,96,7], 'replay_verified': True,
                        'results': [{'model': name, 'mse': 0.5, 'mae': 0.4} for name in config['protocol']['models']]}

            with patch.object(pipeline, 'BASE', root), patch.object(pipeline, 'auth_path', return_value=login), \
                    patch.object(pipeline, 'preflight', return_value={'ready': True}), \
                    patch.object(pipeline.Docker, 'run', self.fake_docker), \
                    patch.object(pipeline, 'run_stage', side_effect=fake_stage):
                pipeline.run_experiment(config, None, 'test-run')
            self.assertEqual([c[0] for c in calls], ['author', 'paper-only'])
            self.assertIsNotNone(calls[0][1])
            self.assertIsNone(calls[1][1])
            manifest = json.loads((root / 'runs/test-run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['author_commit'], 'a'*40)
            self.assertTrue((root / 'runs/test-run/RESULTS.md').is_file())

    def test_stage_failures_are_recorded_and_paper_stage_still_gets_an_independent_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, login, config = self.setup_run(temporary)
            with patch.object(pipeline, 'BASE', root), patch.object(pipeline, 'auth_path', return_value=login), \
                    patch.object(pipeline, 'preflight', return_value={'ready': True}), \
                    patch.object(pipeline.Docker, 'run', self.fake_docker), \
                    patch.object(pipeline, 'run_stage', side_effect=RuntimeError('author training failed')) as stages:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete execution'):
                    pipeline.run_experiment(config, None, 'failed-run')
            self.assertEqual(stages.call_count, 2)
            manifest = json.loads((root / 'runs/failed-run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            report=json.loads((root / 'runs/failed-run/benchmark.json').read_text())
            self.assertEqual(report['outcome'], 'execution_failure')
            self.assertEqual(report['execution_rate'], 0)

    def test_protocol_design_is_a_fresh_stage_before_author_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, login, config = self.setup_run(temporary)
            protocol=config['protocol']
            config['protocol']=None
            order=[]

            def fake_docker_call(instance, command, **kwargs):
                if '-c' in command and 'git' in command[command.index('-c')+1]:
                    order.append('fetch')
                return Orchestration.fake_docker(instance,command,**kwargs)

            def stage(stage, directory, inputs, repo, model, timeout, prefix, login_volume):
                order.append(stage)
                if stage=='design':
                    self.assertEqual(sorted(p.name for p in inputs.iterdir()),['dataset.csv','paper.pdf'])
                    self.assertIsNone(repo)
                    return protocol
                self.assertTrue((inputs/'protocol.json').is_file())
                return {'protocol_sha256':'p','dataset_sha256':'d','prediction_shape':[2785,96,7],
                        'replay_verified':True,'results':[{'model':name,'mse':1,'mae':.5} for name in protocol['models']]}

            with patch.object(pipeline,'BASE',root), patch.object(pipeline,'auth_path',return_value=login), \
                 patch.object(pipeline,'preflight',return_value={'ready':True}), \
                 patch.object(pipeline.Docker,'run',fake_docker_call), \
                 patch.object(pipeline,'run_stage',side_effect=stage):
                pipeline.run_experiment(config,None,'designed-run')
            self.assertEqual(order,['design','fetch','author','paper-only'])


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy required for independent evaluator checks')
class TrustedEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'workspace'
        (self.workspace / 'output').mkdir(parents=True)
        self.evaluator = module('evaluate', 'evaluate.py')
        self.protocol = {'features': ['x'], 'lookback': 3, 'horizon': 2,
                         'split': {'train_end': 10, 'validation_end': 15, 'test_end': 21},
                         'models': ['Linear']}
        # Simple data makes hand-checking train-only standardization possible.
        from datetime import datetime, timedelta
        start = datetime(2020, 1, 1)
        self.dataset = self.root / 'data.csv'
        self.dataset.write_text('date,x\n' + ''.join(f'{start+timedelta(hours=i)},{i}\n' for i in range(21)))

    def tearDown(self):
        self.temp.cleanup()

    def targets(self):
        data, mean, scale = self.evaluator.load_data(self.dataset, self.protocol)
        self.assertEqual(mean.tolist(), [4.5])
        self.assertAlmostEqual(scale[0], np.std(np.arange(10)))
        return np.stack([data[t:t+2] for t in range(15, 20)])

    def test_metrics_include_final_partial_batch_and_use_train_scaler(self):
        truth = self.targets()
        np.save(self.workspace / 'output/Linear.npy', truth + 2)
        result = self.evaluator.evaluate(self.dataset, self.protocol, self.workspace, self.root / 'results')
        self.assertAlmostEqual(result['results'][0]['mse'], 4)
        self.assertAlmostEqual(result['results'][0]['mae'], 2)
        self.assertEqual(result['prediction_shape'], [5, 2, 1])
        self.assertEqual(result['results'][0]['forecast_elements'], 10)

    def test_rejects_wrong_shape_nan_and_object_arrays(self):
        path = self.workspace / 'output/Linear.npy'
        for array in (np.zeros((4, 2, 1)), np.full((5, 2, 1), np.nan),
                      np.full((5, 2, 1), 'bad', dtype=object)):
            np.save(path, array)
            with self.assertRaises(ValueError):
                self.evaluator.evaluate(self.dataset, self.protocol, self.workspace, self.root / 'results')

    def test_fresh_inference_must_match_cached_predictions(self):
        original=self.root/'original'
        (original/'output').mkdir(parents=True)
        truth=self.targets()
        np.save(original/'output/Linear.npy',truth)
        np.save(self.workspace/'output/Linear.npy',truth)
        result=self.evaluator.evaluate(self.dataset,self.protocol,self.workspace,self.root/'results',original)
        self.assertTrue(result['replay_verified'])
        np.save(self.workspace/'output/Linear.npy',truth+1)
        with self.assertRaisesRegex(ValueError,'replay'):
            self.evaluator.evaluate(self.dataset,self.protocol,self.workspace,self.root/'results',original)

    def test_rejects_symlinked_prediction_or_directory(self):
        actual = self.root / 'elsewhere.npy'
        np.save(actual, self.targets())
        link = self.workspace / 'output/Linear.npy'
        link.symlink_to(actual)
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(self.dataset, self.protocol, self.workspace, self.root / 'results')
        link.unlink()
        (self.workspace / 'output').rmdir()
        (self.workspace / 'output').symlink_to(self.root)
        with self.assertRaises(ValueError):
            self.evaluator.regular_file(self.workspace, 'output/elsewhere.npy')


if __name__ == '__main__':
    unittest.main()
