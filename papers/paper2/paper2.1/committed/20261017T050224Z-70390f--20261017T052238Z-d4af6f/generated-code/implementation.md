# Independent paper-only implementation

## Evidence and model mapping

The local PDF was extracted using pypdf; method pages 3–6 and relevant appendix pages 12–18 were read, including appendix A's reconstruction pipeline, B's dataset definitions, C's forecasting comparisons, and D's hourly ETT studies. No repository or outside source was accessed. Only FITS is requested by protocol.json.

models.py implements the complex-frequency interpolation of section 3.2 (page 4, Figure 2) and section 3.3 (pages 4–5). Instance normalization precedes rFFT; the DC bin is removed, low frequencies are retained, and one complex affine layer is shared over all seven channels. Its output is padded with zero high frequencies and an explicit zero DC bin, followed by irFFT and inverse normalization. Channel sharing follows page 5's Weight Sharing paragraph. There is no channel mixing, nonlinear activation, extra layer, or learnable normalization affine transform.

Section 4.1 (pages 5–6) defines output length L+H and eta=1+H/L: here 276 and 276/180. Training supervises all 276 points equally using concatenated backcast inputs and forecast targets, as described in sections 3.2 and 4.1. Validation checkpoint selection uses forecasting MSE over only the 96 horizon points. Appendix D (page 14) explicitly reports both F and B+F variants; B+F was chosen before training.

## Architectural ambiguities and choices

The paper does not give an unambiguous formula mapping harmonic labels to integer cutoff bins. Table 5 (page 8), hourly ETT, lookback 180/horizon 96, second-harmonic setting reports 1053 parameters. We fixed cutoff K=26 and output bins floor(K*276/180)=39 before training: a complex affine layer has (26+1)*39=1053 complex entries, exactly matching that table entry. This is an interpretation of its parameter count, not a claim that the harmonic label uniquely implies these bins. The total is 2106 real scalar degrees of freedom. No cutoff or other architecture search was performed, and no test result influenced these choices.

The retained input bins are explicitly 1 through 26; output bins are 1 through 39. This follows page 5's instruction to reintroduce a zero DC term. Floor rounding of noninteger output frequency length is an assumption. The spectrum includes the normal rFFT Nyquist slot; unused bins are zero. No Nyquist edge issue occurs at this cutoff. PyTorch default backward-normalized FFT is used, and the inverse result is multiplied by eta to compensate for differing transform lengths. This amplitude convention is an implementation assumption consistent with interpolation, rather than an explicit paper equation.

RIN uses each input instance/channel's temporal mean and population variance, with sqrt(var+1e-5); the same statistics reverse the transform. The paper names RIN without specifying epsilon, variance convention, or learnable affine parameters. The above choices are assumptions. The complex affine bias and PyTorch complex Linear default initialization are also assumptions; inclusion of bias is consistent with the table count. No third-party model package is used.

## Protocol and data

All neutral controls come from protocol.json, not from claims about author hyperparameters: feature order HUFL,HULL,MUFL,MULL,LUFL,LULL,OT; seed 42; lookback 180; horizon 96; Adam learning rate 0.001; batch size 64; maximum 50 epochs; patience 5; constant learning rate; shuffled training; no dropped final batches; lowest validation forecasting MSE selection. Adam's omitted controls use PyTorch defaults (betas 0.9/0.999, epsilon 1e-8, zero weight decay). CPU execution uses one intra-op and one inter-op thread, deterministic algorithms, and seeded Python, NumPy and PyTorch. No scheduler or gradient clipping is added.

Population mean/std are calculated in float64 exclusively from rows [0,10452), then standardized data are cast to float32. These statistics and the full protocol are saved in the checkpoint. No timestamps are inputs. All stride-one windows are included. Training starts are 180..10356 (10177 windows); validation starts 10452..13840 (3389); test starts 13936..17324 (3389), inclusive. Thus every target lies wholly within its split; prior split rows supply input context only. Final training batch has one window; validation/test final batches have 61 windows. Test evaluation occurs only after loading the validation-selected checkpoint. Replay loads the saved statistics and trained weights; its inference loop slices input context only and never constructs target arrays.

## Execution and results

Training command (from /workspace):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py train > training.log 2>&1
```

Measured training plus validation/checkpoint-selection duration: 30.136166169 seconds, excluding subsequent test evaluation. Training stopped after epoch 25; epoch 20 was selected, validation forecasting MSE 0.3797576601616334. Full actual epoch records are in training.log and results.json. Best trained weights are checkpoints/FITS.pt. One post-selection test evaluation produced float32 output/FITS.npy, shape [3389,96,7], standardized units, chronological target order. Observed test MSE 0.45816145986948004 and MAE 0.46271691698612394 are informational; independent array-based evaluation is authoritative.

Only generic installed Python/PyTorch/NumPy/pandas libraries are needed for inference. pypdf was used only for paper reading. No unavailable dependency, downloads, remote code, author implementation, or protocol departures were required. Relative to the paper's study, this is the prescribed single fixed-seed protocol, not its five-run/grid-search experiment. Architecture interpretations above limit equivalence to the authors' unpublished implementation details.

The minimal inference manifest is replay.json. It lists run.py, models.py, and checkpoints/FITS.pt only. No predictions or targets are replay inputs. Its command loads the checkpoint and writes /replay-output/FITS.npy without training. Replay was checked from a separate workspace directory containing only manifest files, using the original supplied inputs. The initial exact replay command could not create /replay-output because the current worker root filesystem is read-only there. The same inference command was therefore tested with only its output destination changed to /workspace/replay-test/FITS.npy. It succeeded and reproduced predictions within float32 numerical tolerance (rtol=1e-5, atol=2e-6) (replay-check.log). The manifest retains the required /replay-output destination for the fresh replay container, where that directory is supplied writable. This test destination change is the only execution-environment accommodation; there were no experimental protocol departures. Inference uses NumPy-stacked input batches while evaluation uses DataLoader-stacked tensors; differing memory layouts can produce tiny CPU floating-point differences.
