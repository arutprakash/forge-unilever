import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from suites import answer, parse_score


class SuiteTests(unittest.TestCase):
    def test_point_scores_are_summed_not_llm_total(self):
        text = '\n'.join(['Score: [18/20 points]']*5) + '\nOverall Score: [99/100 points]'
        self.assertEqual(parse_score(text,'mixed')['score'],90)
        self.assertEqual(parse_score('\n'.join(['**Score:** 18/20 points']*5),'mixed')['score'],90)

    def test_missing_duplicate_and_invalid_scores_rejected(self):
        for text in ('[20/20 points]'*4,'[20/20 points]'*6,'[21/20 points]'*5):
            with self.assertRaises(ValueError):
                parse_score(text,'high')

    def test_low_requires_all_categories(self):
        text = 'Overall Structure: 24/25\nModel Details: 21/25\nTraining Details: 23/25\nExperimental Integrity: 22/25'
        self.assertEqual(parse_score(text,'low')['score'],90)
        with self.assertRaises(ValueError):
            parse_score(text.replace('Model Details','Other'),'low')

    def test_incomplete_judge_is_not_a_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'transcript.jsonl'
            event = {'type':'item.completed','item':{'type':'agent_message','text':'result'}}
            path.write_text(json.dumps(event)+'\n')
            with self.assertRaises(RuntimeError):
                answer(path)
            with path.open('a') as stream:
                stream.write(json.dumps({'type':'turn.completed'})+'\n')
            self.assertEqual(answer(path),'result')

    def test_judge_mode_removes_mcp_config(self):
        base = Path(__file__).resolve().parents[1]/'docker'
        config = (base/'config.toml').read_text().split('[mcp_servers.offline]')[0]
        self.assertNotIn('mcp_servers',config)
        self.assertIn('shell_tool = false',config)
        self.assertIn('web_search = "disabled"',config)


if __name__ == '__main__':
    unittest.main()
