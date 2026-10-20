# Independent paper-only implementation

## Sources and equation mapping

Only the supplied paper was consulted. All 18 pages, including Appendix A--C,
were extracted locally using pypdf; methodology pp. 4--7 was read in full,
including equations 1--14, and implementation details pp. 13--14 and complexity
analysis pp. 15--18 were read. No repository, internet, author package, or earlier
stage was consulted. The sole protocol model is CrossGNN.

models.py implements AMSI (Section 3.1, pp. 4--5, eqs. 1--5) via FFT amplitudes,
top five frequencies, period-length average pooling with equal kernel/stride,
and time concatenation. A scalar-to-eight-channel linear embedding expands each
pooled measurement. Section 3.2 (pp. 5--6, eqs. 6--10) is implemented with learned
factor-product temporal scores, ReLU then row softmax, top ceil(10/period)
neighbors in each scale, same-scale adjacent/self nodes, retained-edge
renormalization, message/self concatenation, linear update, activation, and L2
normalization. Section 3.3 (pp. 6--7, eqs. 11--13) uses factor-product variable
scores, positive highest-score edges normalized by their sum, and negative
lowest-score edges weighted by inverse scores normalized separately. Its update
also concatenates message and self, then applies a linear map, activation, and
L2 normalization. Section 3.4 (p. 7, eq. 14) decodes channels to one, then time to
96. Appendix A.3 (pp. 13--14) specifies S=5, K=10, C=8 for small datasets and MSE;
these are used. The paper gives no exact numerical parameter count to match.
This implementation has 27,385 trainable parameters.

## Architectural ambiguities and fixed interpretations

These choices were made before training and were not tuned against either split:

- FFT is a one-sided real FFT per input window, averaged only over variables,
  as stated in eq. 1. DC is excluded because its period is undefined. We retain
  five nonzero frequencies ordered by amplitude. No unpooled original scale is
  appended: eq. 5 explicitly concatenates the S selected scales only.
- Period conversion is integer floor L/f. Distinct frequencies can produce
  identical periods; these remain separate scale blocks. Pooling drops remainder
  rows, following eq. 4's floor length. For real FFT nonzero bins, periods are at
  least 2. FFT ties use PyTorch topk ordering.
- The paper does not specify graph latent rank, graph depth N, activation, MLP
  hidden widths, or variable neighbor counts. Use rank 8 for both graph factors,
  one propagation layer in each GNN, ReLU, and affine maps for the embedding and
  two decoder MLPs. No dropout, residual prediction shortcut, instance
  normalization, timestamp embedding, or extra normalization is added.
- For D=7, use three positive and three negative variable neighbors, including
  self if selected. Positive and negative sets are disjoint; positive selection
  takes priority in tied scores. Thus eq. 12 has no overlapping branches.
- Temporal graph factor rows index concatenated node positions rather than
  original timestamps or frequency IDs. Per-window adaptive periods lead to
  variable L'. The paper does not explain how learned graph and decoder shapes
  accommodate this. Allocate up to S*floor(L/2)=240 positions, slice the graph to
  the current concatenation, and right-pad the channel-decoded sequence to 240
  with zeros before the shared time decoder. Padding is masked from all real
  node aggregation and decoder inputs. This makes predictions independent of
  batch composition. Decoder bias is shared across variables.
- Eq. 10 writes separate sums that could count a node twice when scale-selected
  and trend-selected. We use the union N defined immediately before eq. 9,
  counting each edge once. Eq. 8 includes self (distance <=1), which is retained.
- L2 normalization acts on the channel vector independently at each time and
  variable; numerical epsilon is PyTorch's default. Factors are initialized
  Gaussian with standard deviation 0.1; affine layers use PyTorch defaults.

The paper's p. 5 embedding shape L is treated as L', consistent with eq. 5 and
Section 3.4. These interpretations limit claims of exact architectural identity
with an unavailable author implementation.

## Data, controls, and selection

experiment.py reads the exact protocol feature order HUFL,HULL,MUFL,MULL,LUFL,
LULL,OT. It computes float64 channel means and population standard deviations
from rows [0,10452) only, standardizes, then uses float32 tensors. Statistics are
stored inside the checkpoint for inference. Inputs have length 96 and targets
have horizon 96. Stride is one. Train starts range 96..10356 (10,261 windows),
validation starts 10452..13840 (3,389), and test starts 13936..17324 (3,389).
Each end boundary is exclusive. Validation/test inputs use preceding context;
no target crosses a boundary. No timestamp covariates are used.

Python, NumPy and PyTorch seeds are 42. Use one CPU thread and zero loader
workers; training alone shuffles. Adam uses protocol learning rate 0.001,
constant throughout, batch size 64, maximum 50 epochs, patience 3, lowest
validation MSE checkpoint selection. All final partial batches are included;
MSE is aggregated by element count, including validation/test. Unspecified Adam
options use PyTorch defaults (betas 0.9/0.999, epsilon 1e-8, weight decay zero).
Test evaluation occurs only after training stops and the selected checkpoint is
loaded. No model or hyperparameter was changed after test evaluation.

These are neutral protocol controls, not a claim that all were specified by the
paper. In particular no learning-rate grid search from Appendix A.3 was run.
There are no departures from the supplied experimental protocol.

## Execution, dependencies, and results

Executed `python experiment.py > training.log 2>&1`. The selected checkpoint is
best_CrossGNN.pt. training_metadata.json and training.log record each epoch.
Measured optimization plus validation/checkpoint time: 22.97418 seconds, five
epochs; selected epoch 2, validation MSE 0.6292938325. Test predictions have
shape [3389,96,7], float32, standardized units. Self-reported test MSE is
0.9178035788 and MAE 0.6800987543; independent pipeline metrics are authoritative.
The relatively high error is reported as observed, without test-driven tuning.

Dependencies are generic installed Python, PyTorch, NumPy and pandas. pypdf was
used only to read the paper. This CPU implementation uses dense masked adjacency
matrices; it implements the restricted-edge semantics but does not reproduce the
TVM sparse linear-complexity optimization discussed in Section 4.6 and Appendix
C. No TVM or GPU is available/required. This is a computational implementation
limitation, not a protocol change.

## Inference replay verification

replay.json declares models.py, experiment.py, and best_CrossGNN.pt only. All are
regular workspace-relative files; no stored predictions or target arrays are
listed. Checkpoint includes weights, train-only scaling statistics, protocol and
selected epoch. `--predict` loads the saved weights and performs no training.

Copied just these files into /workspace/replaycheck and executed the inference
command there with the original inputs. The first attempt to use /replay-output
failed because this worker mounts root read-only. Repeated with only the output
argument changed to /workspace/replaycheck/output/CrossGNN.npy. The command
succeeded and its predictions were bitwise identical to output/CrossGNN.npy;
shape, float32 dtype, and finiteness assertions passed. replay.json keeps the
required /replay-output/CrossGNN.npy destination for the fresh replay container,
where the pipeline supplies that writable directory. All requested training,
checkpointing, prediction, and replay verification work completed.
