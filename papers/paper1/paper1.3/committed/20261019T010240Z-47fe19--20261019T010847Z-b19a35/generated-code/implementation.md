# Paper-only implementation

Source: /inputs/paper.pdf, Are Transformers Effective for Time Series Forecasting?, arXiv:2205.13504v3. Extracted locally using pypdf; relevant main text section 4 (pp. 3–4), Figure 2 (p. 4), and appendix B–D (pp. 9–11), including D.3 continuation on p. 11, were read in full. No repository or external material was consulted.

## Definitions and architectural interpretation

- Linear: section 4, p. 4: y_i = W x_i, W in R^(T x L), shared across channels. No cross-channel operations, embeddings, activations, or dropout.
- NLinear: section 4, p. 4 and appendix D.1, p. 10: subtract each channel's last context value, apply shared temporal W, then add that value to every horizon position.
- DLinear: section 4/Figure 2, p. 4: compute moving-average trend, subtract it from input for remainder, apply separate shared temporal weights, and sum their forecasts. Appendix B.2, p. 9 explicitly specifies kernel 25.
- Biases are omitted to follow both the explicit equation and appendix B.2 counts TL, TL, 2TL. Counts here are 32,256, 32,256 and 64,512; code asserts them. Table 8, p. 8 reports 139.7K for L=96,T=720, which is slightly inconsistent with the exact 2TL count (138,240). We prioritize the exact formula rather than infer additional parameters.
- Moving-average boundary handling is unspecified: assume centered average with 12 endpoint replications at each side, keeping length L and never accessing future targets.
- Initialization is unspecified for the main forecasting experiment: use PyTorch Linear's default uniform random weights (range +/-1/sqrt(L)), independently for each branch. Appendix D.3, p. 11 uses 1/L initialization specifically to smooth visualization weights; that visualization-specific choice is not applied here. This is an assumption, not a claim about authors' implementation.

## Fixed neutral protocol and data

The protocol supplies model names, feature order HUFL,HULL,MUFL,MULL,LUFL,LULL,OT, L=336,T=96, seed=42, train_end=8640, validation_end=11520,test_end=14400. These are experimental controls, not inferred author hyperparameters. Fit float64 mean and population std solely on rows [0,8640), then transform to float32. Reject nonfinite values and zero scales; no imputation. Ignore timestamps as model inputs.

Stride-one target starts: training 336..8544 inclusive (8209 windows), validation 8640..11424 (2785), test 11520..14304 (2785). Every target is fully inside its split; contexts use preceding rows. Outputs are float32 [window,96,7] in increasing target-start order, standardized units. Partial final batches are retained.

Train independently with reset seed 42, one CPU thread, deterministic algorithms, Adam lr=0.001 with default betas=(0.9,0.999), eps=1e-8, weight_decay=0, no clipping, constant schedule, batch 64, shuffled training, max 50 epochs, patience 5. Minimize equally weighted MSE across all target elements. Validation MSE is accumulated by element, including partial batches. Strict improvement selects the checkpoint (first epoch on ties); stop after five consecutive non-improving epochs. Test evaluation occurs only after loading that model's validation-selected checkpoint. No hyperparameters or architectures were changed based on test results.

## Commands and dependencies

Repeat everything: `python run.py`. Individual runs: `python experiment.py train --model Linear` (substitute NLinear or DLinear). Training stdout/stderr is redirected by run.py to per-model logs. Inference commands appear verbatim in replay.json. Checkpoints contain learned weights, normalization statistics, protocol and selected epoch; no input windows, targets or predictions are cached in checkpoints.

Only installed generic Python/PyTorch/NumPy/pandas libraries are imported for inference/training; pypdf was used for reading. No dependency limitations prevented these architectures, and no model implementation package, network access, or author source was used. Protocol departures: none. Boundary handling, initialization, and unspecified Adam defaults are documented assumptions. All requested runs completed.

## Observed training results

Durations below measure the training/validation/checkpoint loop using perf_counter, excluding process startup and final test evaluation; see logs and output metadata. Metrics are local checks, not authoritative pipeline scores.

| Model | Seconds | Epochs run | Selected epoch | Validation MSE | Test MSE | Test MAE |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 9.068965 | 21 | 16 | 0.64564650 | 0.37838228 | 0.40194979 |
| NLinear | 3.949194 | 9 | 4 | 0.67658385 | 0.38120724 | 0.40178957 |
| DLinear | 15.767025 | 12 | 7 | 0.64413008 | 0.38317105 | 0.40781398 |

Replay verification: copied only manifest-listed sources and checkpoints to /workspace/replay_check and ran all three inference commands there. The training worker forbids creating /replay-output (read-only root), so the local smoke test substituted only --output with /workspace/replay_check/predictions/<model>.npy. All three results were finite float32 [2785,96,7] and bitwise identical to training outputs. The manifest retains the required /replay-output destination, which the fresh replay container supplies. This is a local verification path limitation, not a training protocol departure.
