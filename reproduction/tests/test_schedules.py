import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import validate_protocol


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.protocol = validate_protocol(json.loads((Path(__file__).resolve().parents[1]/'presets/dlinear-etth1.json').read_text())['protocol'])

    def test_existing_constant_protocol_unchanged(self):
        self.assertEqual(validate_protocol(self.protocol), self.protocol)

    def test_delayed_exponential_is_preserved(self):
        self.protocol['training']['schedule'] = {'type':'exponential','factor':0.8,'start_epoch':4}
        self.assertEqual(validate_protocol(self.protocol), self.protocol)

    def test_invalid_schedules_rejected_precisely(self):
        bad = ['decay every epoch', {'type':'cosine','factor':0.8,'start_epoch':4},
               {'type':'exponential','factor':True,'start_epoch':4},
               {'type':'exponential','factor':float('nan'),'start_epoch':4},
               {'type':'exponential','factor':0,'start_epoch':4},
               {'type':'exponential','factor':1.1,'start_epoch':4},
               {'type':'exponential','factor':0.8,'start_epoch':False},
               {'type':'exponential','factor':0.8,'start_epoch':0},
               {'type':'exponential','factor':0.8,'start_epoch':4,'code':'unsafe'}]
        for schedule in bad:
            with self.subTest(schedule=schedule):
                p = copy.deepcopy(self.protocol); p['training']['schedule'] = schedule
                with self.assertRaisesRegex(ValueError, 'schedule'):
                    validate_protocol(p)

    def test_checkpoint_error_is_distinct(self):
        self.protocol['training']['checkpoint_selection'] = 'test minimum'
        with self.assertRaisesRegex(ValueError,'checkpoint_selection'):
            validate_protocol(self.protocol)
