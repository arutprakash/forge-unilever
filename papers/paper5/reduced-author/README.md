# Reduced-size FEDformer author run

Run from the repository root:

```sh
python3 papers/paper5/reduced-author/run.py --smoke
python3 papers/paper5/reduced-author/run.py
```

The first command checks one actual training batch (forward, backward, Adam
step) for each variant and records parameter counts and timings. It does not
save trained checkpoints. The second trains both variants with the original
frozen data/training protocol and validation checkpoint selection. It may take
hours; logs append under `papers/paper5/reduced-author/runs/small-v1/`.
Repeat the training command to resume interrupted training from segment
checkpoints (up to about 450 seconds of progress may be lost on interruption).
Do not run multiple commands against the same workspace concurrently.

The separate runner keeps the central pipeline and the original sealed archive
unchanged. It copies author code and the wrapper from the retained paper5.1
run, including its existing Fourier optimization. That local archive and its
Docker runtime image must remain available. No generated implementation is used.
Outputs are exploratory author results, not a sealed Forge benchmark or an
automatic comparison against the earlier paper-only run.

Architecture changes: `d_model=64`, `d_ff=256`, wavelet self/cross channels `c=8`
(originally 512, 2048, 128/64). Eight heads, wavelet degree 8, two encoder layers,
one decoder layer, modes, dropout, and decomposition remain as in the retained
author wrapper. The data splits, batch size, seed, learning rate, epoch cap,
patience, and checkpoint selection are unchanged. Full optimizer batches replace
the failed run's emergency wavelet microbatch workaround. This is a reduced-size
author configuration, not reproduction of the published configuration and not
an architecture-matched comparison with the paper-only model.

Docker is offline, limited to 4 GB and two CPUs by default. Override those limits
for this separate run with `--memory 8g --cpus 4` if needed. Docker Desktop must
have sufficient resources allocated. Commands here have no 600-second bridge
timeout or two-hour stage timeout. The underlying training segments still save
progress around 450 seconds, checked at optimizer-batch boundaries.

Only `runs/small-v1` is used; changing runner source after preparation does not
rewrite this existing workspace. Use a new workspace name in the runner for a
different architecture, so incompatible checkpoints cannot be silently reused.
