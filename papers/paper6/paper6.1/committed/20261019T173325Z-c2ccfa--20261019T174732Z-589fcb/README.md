# paper6.1 — preserved experiment

Source run: `20261019T173325Z-c2ccfa`. Generated source is copied without modification.
See [numerical results](results/RESULTS.md), [protocol](protocol.json), and
[provenance](provenance.json). See [alignment evaluation](benchmark/RESULTS.md).

## Reproduce from the repository root

```bash
python3 forge.py run \
  --paper PATH_TO_ORIGINAL_PDF \
  --dataset PATH_TO_ORIGINAL_CSV \
  --author-repo https://github.com/hqh0728/CrossGNN.git \
  --author-ref 0407abd085ee8342abe0bbe6de5b2ab17c44373c \
  --protocol papers/paper6/paper6.1/committed/20261019T173325Z-c2ccfa--20261019T174732Z-589fcb/protocol.json \
  --experiment paper6.1 \
  --benchmark-suite autoreproduce
```

Use the original input files matching the SHA-256 hashes in provenance. Runtime
dependencies and image identities are recorded there; the pipeline Dockerfile
provides the pinned environment. Code entry points are documented in
`generated-code/README.md`. Paths such as `/inputs` refer to container mounts.
Checkpoints are excluded, so inference replay requires retraining or the original
ignored run archive. Exact rerun scores may vary with model generation and training.

## Verify this export

```bash
python3 forge.py verify --run-dir papers/paper6/paper6.1/committed/20261019T173325Z-c2ccfa--20261019T174732Z-589fcb
```

Only small text source and reports are preserved. Original runs retain large
artifacts and raw audit traces. Benchmark alignment uses upstream evaluators on a
custom task and is not an official ReproduceBench leaderboard score. Generated
code may require license review before redistribution; no new license is assigned.
