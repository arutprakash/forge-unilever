# Authors' ETTh1 results

Run from the repository root:

```sh
python3 -m venv --system-site-packages paper1/.venv
paper1/.venv/bin/python -m pip install 'numpy<2'
paper1/.venv/bin/python paper1/run_author_results.py
```

The runner executes the existing `author-code/LTSF-Linear/run_longExp.py` entry point without source changes for DLinear, NLinear, and Linear. It uses the authors' ETTh1 script configuration: multivariate (7 channels), lookback 336, horizon 96, batch 32, learning rate 0.005, seed 2021, up to 10 epochs, early stopping patience 3. All three models use the same configuration. CPU execution uses one thread and zero data-loader workers.

The author loader fits StandardScaler on the first 8,640 rows, validates on the next 2,880, and tests on the next 2,880, with lookback context included. There are 8,209 training, 2,785 validation, and 2,785 test windows. Test `drop_last=False` is preserved as present in this checkout. Metrics are in standardized units across all seven channels. Checkpoints are selected by validation loss.

`summary.csv` and `summary.json` contain final metrics. Each model directory contains `run.log`, `result.txt`, the best checkpoint under `checkpoints/`, forecast arrays under `results/`, and example forecast plots under `test_results/`. Large binary outputs and the environment are ignored by Git but retained locally.

The environment uses installed PyTorch 2.2.2 with NumPy 1.26.4; the authors' requirements specify PyTorch 1.9.0. This is a single-seed run, so differences from the paper's rounded reference values should not be interpreted as exact reproduction failures.

| Model | MSE | MAE |
|---|---:|---:|
| DLinear | 0.384144 | 0.404713 |
| NLinear | 0.373704 | 0.394224 |
| Linear | 0.374749 | 0.396704 |
