import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import seal_run, verify_run
from exporter import export_run


class ExportTests(unittest.TestCase):
    def fixture(self, base):
        run = Path(base)/'experiment/runs/test'
        files = {'manifest.json':json.dumps({'status':'complete','experiment':'paper1.3',
            'config':{'author':{'url':'https://github.com/example/model','ref':'HEAD'}},
            'author_commit':'a'*40,'inputs':{},'images':{}}),
            'inputs/protocol.json':'{}','paper-only/generated-code/models.py':'# generated\n',
            'paper-only/generated-code/checkpoints/secret.py':'# excluded\n',
            'paper-only/generated-code/output/pred.npy':'large binary placeholder',
            'author/repo/model.py':'# author excluded\n',
            'author/run/results/metrics.json':'{}','paper-only/results/metrics.json':'{}'}
        files.update({name:'{}' for name in ('RESULTS.md','benchmark.json','comparison.json','benchmark-policy.json')})
        for name,value in files.items():
            path = run/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(value)
        seal_run(run)
        return run

    def test_export_is_small_verifiable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as base:
            run = self.fixture(base)
            original = verify_run(run)
            target = export_run(run)
            self.assertTrue((target/'generated-code/models.py').is_file())
            self.assertFalse((target/'author').exists())
            self.assertFalse((target/'generated-code/checkpoints').exists())
            self.assertFalse((target/'generated-code/output').exists())
            verify_run(target)
            self.assertEqual(export_run(run),target)
            self.assertEqual(verify_run(run),original)
            (target/'generated-code/models.py').write_text('changed')
            with self.assertRaises(ValueError):
                export_run(run)

    def test_failed_run_and_wrong_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as base:
            run = self.fixture(base)
            evaluation = Path(base)/'evaluation'; evaluation.mkdir()
            (evaluation/'suite-results.json').write_text(json.dumps({'status':'complete','source_run':str(run),
                'source_seal':{'seal_sha256':'wrong'}}))
            seal_run(evaluation)
            with self.assertRaises(ValueError):
                export_run(run,evaluation)
            path = run/'manifest.json'; value=json.loads(path.read_text()); value['status']='partial'
            path.write_text(json.dumps(value)); seal_run(run)
            with self.assertRaises(ValueError):
                export_run(run)


if __name__ == '__main__':
    unittest.main()
