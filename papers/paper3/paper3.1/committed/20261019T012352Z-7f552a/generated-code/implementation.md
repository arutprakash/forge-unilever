# Independent paper-only implementation

Source: local PDF, An Analysis of Linear Time Series Forecasting Models, Toner and Darlow. Extracted with pypdf into paper.txt. Read sections 3.1, 3.3, 4, 5 and relevant appendix D.2/D.2.1, E and F in full. No external sources, model packages or author code used.

## Model mapping and architecture

OLS implements the affine forecasting class of Definition 3.1 (p.2), with the least-squares solution in Definition D.1, equation 14 (p.18). Appendix E (p.19) specifies SVD-based LinearRegression; this independent implementation uses scipy.linalg.lstsq LAPACK GELSD directly with a constant design column. Predict xW+b. Stored coefficients: (720+1)*96 = 69,216, shared across all seven channels. No cross-channel predictors.

OLS_IN maps the protocol spelling to paper OLS+IN (Table 1, p.5; sections 4/5, pp.6–8). Section 3.3, Definition 3.4 and Lemma 3.5 (pp.3–4) specify the class. Appendix D.2.1 (p.19) gives its closed-form construction: for each context x, compute temporal mean m and population standard deviation s; regress y-m on [x-m,s], without intercept. Forecast m + (x-m)W + sb. This minimizes error in globally standardized units, rather than weighting samples by inverse instance variance. Centering implicitly enforces the effective temporal coefficients summing to one. Stored parameters are again 69,216; the temporal constant direction is redundant, leaving 69,120 independent parameters. No RevIN affine parameters.

## Choices and numerical assumptions

Section 3.1 excludes cross-channel information; the paper does not unambiguously specify whether OLS coefficients are shared or separate per channel. We pool window/channel pairs to fit one shared mapping; p.20 F.2 explicitly discusses shared temporal weights. All channels retain their protocol order. Neither multivariate concatenation nor per-channel tuning is used.

Population instance standard deviation is assumed. We use the exact sigma-augmented class from D.2.1, with no epsilon: there is no division by sigma and zero-variance instances are well-defined. This resolves the unspecified epsilon in Definition 3.4 using the closed-form equations.

SVD singular values below max(design.shape)*float64 epsilon times the largest singular value are discarded (standard dimension-scaled numerical rank convention). This handles the centered design's mathematically redundant direction. An initial diagnostic run used SciPy's default cutoff, which incorrectly retained that direction at floating-point precision. It was replaced for structural numerical reasons, with both fits rerun. No model choice or hyperparameter was selected from test performance. Final authoritative log is training.log; earlier diagnostic logs were removed. Final ranks: OLS 721, OLS_IN 720. No regularization. All arrays and fitting use float64.

## Protocol and execution

Exactly the protocol model names, lookback 720, horizon 96, seed 42, feature order, and split endpoints 8545/11426/14307 are used. Per-channel mean and population std use only rows [0,8545). Later rows never affect fitting. Train target starts 720..8449, validation 8545..11330, test 11426..14211 (inclusive); 7730/2786/2786 windows. All target intervals stay inside their split, and prior-split input context is allowed. Stride one throughout. Test results are generated only after saving the validation-selected checkpoint.

Protocol optimizer OLS is honored with one exact solve (recorded as epoch 1). There is one candidate per model; finite validation MSE selects it. Learning rate, 50-epoch cap, constant schedule and patience 5 have no operational meaning for a deterministic closed-form optimizer and are not converted into iterative Adam training. The training sample order is shuffled using seed 42. All samples participate in the solve; batch size 64 governs validation and inference, including the final partial batch (34 windows). The fit is global, not minibatch optimization. These are the only adaptations of iterative training controls. No other protocol departures.

One CPU numerical thread is enforced before NumPy/SciPy imports. Dependencies are installed generic Python 3.11, NumPy, pandas and SciPy; no model-specific dependency or download. The pypdf dependency was used only for research extraction. No dependency limitation prevented completion. Source is self-contained for inference; generic installed libraries suffice.

Repeat command: `python experiment.py train > training.log 2>&1`. This fits both models, evaluates validation, saves checkpoints, then evaluates test. Inference commands are in README.md and replay.json. Checkpoints store only coefficients, scaler, model identity and dimensions. They do not contain training samples, target arrays or predictions.

## Final measured runs

OLS: fit 5.034660 seconds; fit plus validation/checkpoint/test 5.757195 seconds. Validation-selected epoch 1 (one exact solve), validation MSE 0.671852119854.

OLS_IN: fit 5.527418 seconds; fit plus validation/checkpoint/test 6.489973 seconds. Validation-selected epoch 1 (one exact solve), validation MSE 0.688147467616.

Inference replay tested from a separate directory containing only manifest-listed source/checkpoints, using original inputs. Both arrays matched bit for bit, finite float64 [2786,96,7]. This worker does not allow creating /replay-output (read-only root); the initial literal command failed only at output-directory creation. Tests repeated with only --output changed to writable /workspace/replay_check/<model>.npy. The manifest retains the required /replay-output paths for the fresh replay container.
