# Paper-derived LTSF-Linear implementation

Independent implementation of **Are Transformers Effective for Time Series
Forecasting?**, using the local `paper1/paper1.pdf` (arXiv v3, August 2022).
The authors' local/remote code was not consulted or imported during this task.
The same assistant had seen author-code details in the preceding task, so this
is not a strict clean-room provenance claim. No claim of exact reproduction is made.

## Model specification from the paper

- Section 4: Linear applies a shared temporal matrix `W` to each variate,
  without mixing channels. NLinear subtracts each channel's final observation,
  applies that matrix, and adds the final observation back.
- Section 4: DLinear decomposes the input into a moving-average trend and its
  remainder, applies a separate temporal matrix to each, then sums the forecasts.
- Appendix B.2: use a 25-point moving average and lookback 336; the paper lists
  `T*L` parameters for Linear/NLinear and `2*T*L` for DLinear. Consequently this
  implementation has **no biases** (32,256 / 64,512 parameters at horizon 96).
- Sections 5.1 and Table 2: evaluate MSE/MAE with horizon 96, across seven ETTh1
  variates; Repeat is the final-observation baseline.
- Appendix D.1 discusses standardization by training means and variances.
- Appendix D.3 uses constant initialization specifically for interpretability
  plots. That visualization-specific choice is not used for these runs.

## Choices not specified in the paper

Appendix B.2 delegates additional hyperparameters to the code, which is outside
this task's allowed sources. The following are explicit implementation choices,
not claimed to be paper settings or values recovered from the authors' code:

| Choice | Value |
|---|---|
| Chronological split | 60% train, 20% validation, 20% test of all 17,420 rows |
| Split row boundaries, zero-based | train `[0,10452)`, validation `[10452,13936)`, test `[13936,17420)` |
| Standardization | Per channel, population standard deviation, training rows only |
| Window stride | 1; input context may precede a split, all target rows stay within it |
| Moving-average boundaries | Centered average, replicate endpoint padding |
| Initialization | Uniform random weights in `[-1/sqrt(L), 1/sqrt(L)]` |
| Training objective | Mean squared error over all forecast steps and channels |
| Optimizer | Adam, constant learning rate 0.001, default betas/epsilon, no weight decay |
| Batch size | 64, include partial final batches |
| Training budget | At most 50 epochs; stop after 5 epochs without validation MSE improvement of at least `1e-6` |
| Checkpoint selection | Lowest qualifying validation MSE; test evaluated only after selection |
| Random seed | 42, reset for each model; separate seeded training shuffle generator |
| Execution | Deterministic PyTorch operations, CPU, one thread, zero loader workers |

These choices use a different split and training protocol from the preceding
author-code experiment. **Do not treat the numerical difference between those
runs as an implementation-quality comparison.** Paper Table 2 numbers also
cannot be claimed as directly reproduced under this independently chosen protocol.
There is no hyperparameter search or test-set tuning.

## Run

From the repository root, using the already available local environment:

```sh
paper1/.venv/bin/python paper1/generated-code/test_paper_properties.py
paper1/.venv/bin/python -u paper1/generated-code/run.py
```

For a separate environment, install `requirements.txt` into a virtualenv.
All imports are standard libraries, listed packages, or files in this folder.
No author-code directory is added to Python's search path. The models,
preprocessing, training loop, evaluation, and checkpoint handling are all here.

## Outputs

`results/etth1-336-96/summary.csv` contains aggregate standardized MSE/MAE.
`manifest.json` records configuration, split times, scaler values, package
versions, and SHA-256 hashes of the dataset, paper, and implementation sources.
Each model has `metrics.json`, `training.json`, and `predictions.npy` with axes
`[test_window, forecast_step, channel]`; learned models also have `best.pt`.
Shared `targets.npy` and `test_target_start_rows.npy` allow metric verification
and mapping predictions to original timestamps. Undo scaling with the recorded
channel standard deviations and means to recover original units. Binary arrays
and checkpoints are retained locally and ignored by Git.

The property checks verify the paper's parameter counts, channel independence,
NLinear translation equivariance, constant preservation in decomposition,
forecast boundaries, and correct metrics for a partial final batch.

## Completed ETTh1 run

| Model | MSE | MAE | Best epoch | Epochs run |
|---|---:|---:|---:|---:|
| Linear | 0.427660 | 0.441944 | 20 | 25 |
| NLinear | 0.434310 | 0.447443 | 8 | 13 |
| DLinear | 0.428687 | 0.443430 | 14 | 19 |
| Repeat | 1.655852 | 0.845358 | n/a | 0 |

Each result covers 3,389 test windows, 96 steps, and seven channels (2,277,408 forecast elements). Full-array recomputation matches the recorded metrics; saved checkpoint forecasts match the first and last prediction windows. See `verification.json`. All five property tests passed. Training console output is in `run.log`.
