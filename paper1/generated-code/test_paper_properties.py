"""Check mathematical properties in the paper and split leakage boundaries."""
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ForecastWindows
from models import DLinear, Linear, NLinear
from run import evaluate


class PaperProperties(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_parameter_counts_and_no_channel_mixing(self):
        history = torch.randn(2, 31, 7)
        order = [3, 1, 0, 6, 2, 5, 4]
        for cls, multiplier in ((Linear, 1), (NLinear, 1), (DLinear, 2)):
            model = cls(31, 8)
            self.assertEqual(sum(p.numel() for p in model.parameters()), multiplier * 31 * 8)
            torch.testing.assert_close(model(history[:, :, order]), model(history)[:, :, order])
            changed = history.clone()
            changed[:, :, 0] += 100
            torch.testing.assert_close(model(changed)[:, :, 1:], model(history)[:, :, 1:])

    def test_nlinear_translation_equivariance_and_constants(self):
        model = NLinear(31, 8)
        history = torch.randn(2, 31, 7)
        shift = torch.randn(2, 1, 7)
        torch.testing.assert_close(model(history + shift), model(history) + shift)
        constant = shift.expand(-1, 31, -1)
        torch.testing.assert_close(model(constant), shift.expand(-1, 8, -1))

    def test_decomposition_preserves_constants_and_reconstructs(self):
        model = DLinear(31, 8)
        history = torch.randn(2, 31, 7)
        trend, remainder = model.decompose(history)
        torch.testing.assert_close(trend + remainder, history)
        constant = torch.full_like(history, 3)
        trend, remainder = model.decompose(constant)
        torch.testing.assert_close(trend, constant)
        torch.testing.assert_close(remainder, torch.zeros_like(constant))

    def test_forecast_targets_stay_inside_split(self):
        values = np.arange(100, dtype=np.float32).reshape(-1, 1)
        windows = ForecastWindows(values, 60, 80, 12, 5)
        self.assertEqual(len(windows), 16)
        first_x, first_y = windows[0]
        self.assertEqual(first_x[:, 0].tolist(), list(range(48, 60)))
        self.assertEqual(first_y[:, 0].tolist(), list(range(60, 65)))
        _, last_y = windows[-1]
        self.assertEqual(last_y[-1].item(), 79)

    def test_metrics_include_partial_last_batch(self):
        from models import Repeat
        windows = ForecastWindows(np.arange(17, dtype=np.float32).reshape(-1, 1), 7, 17, 3, 2)
        metrics, predictions, targets = evaluate(Repeat(3, 2), DataLoader(windows, batch_size=4), True)
        self.assertEqual(len(predictions), 9)
        self.assertEqual(metrics['elements'], 18)
        self.assertAlmostEqual(metrics['mse'], 2.5)
        self.assertAlmostEqual(metrics['mae'], 1.5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
