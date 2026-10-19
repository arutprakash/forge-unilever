import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import Journal, seal_run, verify_run, verify_chain
from benchmark import score, usage
from configuration import validate_protocol, from_inputs
import pipeline

BASE = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((BASE/'benchmark-policy.json').read_text())
        self.reference = {'protocol_sha256':'p', 'dataset_sha256':'d', 'prediction_shape':[5,2,1],
                          'replay_verified':True, 'results':[{'model':'A','mse':1.0,'mae':0.5}]}
        self.stages = {'author':{'status':'evaluated'},'paper-only':{'status':'evaluated'}}

    def test_exact_tolerance_boundary_and_mismatch(self):
        generated = copy.deepcopy(self.reference)
        generated['results'][0]['mse'] = 1.05
        report = score(self.reference, generated, self.policy, self.stages)
        self.assertEqual(report['outcome'], 'numerical_match')
        generated['results'][0]['mae'] = 0.526
        report = score(self.reference, generated, self.policy, self.stages)
        self.assertEqual(report['outcome'], 'metric_mismatch')

    def test_changed_dataset_protocol_or_cached_arrays_cannot_pass(self):
        for key, value in [('dataset_sha256','other'),('protocol_sha256','other'),('replay_verified',False)]:
            candidate = copy.deepcopy(self.reference)
            candidate[key] = value
            with self.assertRaises(ValueError):
                score(self.reference, candidate, self.policy, self.stages)

    def test_zero_reference_and_failure_are_not_fabricated_success(self):
        a = copy.deepcopy(self.reference)
        a['results'][0]['mse']=0
        report=score(a,self.reference,self.policy,self.stages)
        self.assertIsNone(report['models'][0]['metrics']['mse']['relative_gap'])
        report=score(None,self.reference,self.policy,{'author':{'status':'failed'},'paper-only':{'status':'evaluated'}})
        self.assertEqual(report['execution_rate'],.5)
        self.assertEqual(report['outcome'],'execution_failure')

    def test_usage_missing_is_null_and_actual_turns_are_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'events.jsonl'
            self.assertIsNone(usage(p)['tokens'])
            p.write_text('not JSON\n'+json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}})+'\n')
            self.assertEqual(usage(p)['tokens']['input_tokens'],10)
            self.assertIsNone(usage(p)['usd'])


class AuditTests(unittest.TestCase):
    def test_chain_and_file_seal_detect_edits_and_extra_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            journal=Journal(root/'audit')
            journal.emit('start', access_token='secret-value')
            journal.emit('finish', code=0)
            self.assertNotIn('secret-value',journal.path.read_text())
            self.assertEqual(verify_chain(journal.path)[0],2)
            (root/'result.json').write_text('{}')
            seal_run(root)
            self.assertEqual(verify_run(root)['verified_files'],2)
            (root/'extra.py').write_text('changed=True')
            with self.assertRaises(ValueError): verify_run(root)
            (root/'extra.py').unlink()
            text=journal.path.read_text().replace('"code": 0','"code": 1')
            journal.path.write_text(text)
            with self.assertRaises(ValueError): verify_chain(journal.path)
            with self.assertRaises(ValueError): verify_run(root)


class ReplayAndConfigTests(unittest.TestCase):
    def test_image_recipe_fingerprint_ignores_host_bytecode_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'docker').mkdir()
            (root/'docker/Dockerfile').write_text('FROM python:3.11')
            with patch.object(pipeline,'BASE',root):
                pipeline.configure_images({'codex_version':'0.160.1'})
                first=dict(pipeline.IMAGES)
                (root/'docker/__pycache__').mkdir()
                (root/'docker/__pycache__/host.pyc').write_bytes(b'host-only-bytecode')
                pipeline.configure_images({'codex_version':'0.160.1'})
                self.assertEqual(first,pipeline.IMAGES)
    def test_replay_snapshot_has_only_declared_sources_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); workspace=root/'work'; workspace.mkdir()
            (workspace/'predict.py').write_text('print("test")')
            (workspace/'weights.npy').write_bytes(b'checkpoint')
            (workspace/'output').mkdir(); (workspace/'output/A.npy').write_bytes(b'predictions')
            declaration={'files':['predict.py','weights.npy'],'commands':[{'model':'A','argv':['python','predict.py']}]}
            (workspace/'replay.json').write_text(json.dumps(declaration))
            pipeline.prepare_replay(workspace,root/'snapshot',{'models':['A']})
            self.assertFalse((root/'snapshot/output').exists())
            declaration['files'].append('output/A.npy')
            (workspace/'replay.json').write_text(json.dumps(declaration))
            with self.assertRaises(ValueError):
                pipeline.prepare_replay(workspace,root/'bad',{'models':['A']})

    def test_protocol_rejects_metadata_code_and_path_injection(self):
        protocol=json.loads((BASE/'presets/dlinear-etth1.json').read_text())['protocol']
        for change in ({'notes':'fetch external code'}, {'models':['../escape']}, {'horizon':-1}):
            candidate={**protocol,**change}
            with self.assertRaises(ValueError): validate_protocol(candidate)
        generic={**protocol,'models':['FITS'],'features':['0','temperature (C)']}
        self.assertEqual(validate_protocol(generic)['models'],['FITS'])

    def test_three_inputs_auto_selects_preset_without_mounting_reference(self):
        repo=BASE.parent
        config=from_inputs(repo/'papers/paper1/paper1.pdf',repo/'datasets/ETTh1.csv','https://github.com/cure-lab/LTSF-Linear')
        self.assertEqual(config['preset'],'dlinear-etth1-336-96')
        trailing=from_inputs(repo/'papers/paper1/paper1.pdf',repo/'datasets/ETTh1.csv','https://github.com/cure-lab/LTSF-Linear/')
        self.assertEqual(trailing['preset'],config['preset'])
        self.assertNotIn('published_reference',config['protocol'])
        self.assertNotIn('github',json.dumps(config['protocol']))
        other=from_inputs(repo/'papers/paper1/paper1.pdf',repo/'datasets/ETTh1.csv','https://github.com/example/new-method')
        self.assertIsNone(other['protocol'])


if __name__ == '__main__':
    unittest.main()
