# paper3.1 — preserved experiment

Source run: `20261019T012352Z-7f552a`. Generated source is copied without modification.
See [numerical results](results/RESULTS.md), [protocol](protocol.json), and
[provenance](provenance.json). No alignment evaluation is included.

## Reproduce from the repository root

```bash
python3 forge.py run \
  --paper PATH_TO_ORIGINAL_PDF \
  --dataset PATH_TO_ORIGINAL_CSV \
  --author-repo https://github.com/sir-lab/linear-forecasting.git \
  --author-ref 9abf0db2b73ff27ebdaf8b852d9eca0fed660b71 \
  --protocol papers/paper3/paper3.1/committed/20261019T012352Z-7f552a/protocol.json \
  --experiment paper3.1
```

Use the original input files matching the SHA-256 hashes in provenance. Runtime
dependencies and image identities are recorded there; the pipeline Dockerfile
provides the pinned environment. Code entry points are documented in
`generated-code/README.md`. Paths such as `/inputs` refer to container mounts.
Checkpoints are excluded, so inference replay requires retraining or the original
ignored run archive. Exact rerun scores may vary with model generation and training.

## Verify this export

```bash
python3 forge.py verify --run-dir papers/paper3/paper3.1/committed/20261019T012352Z-7f552a
```

Only small text source and reports are preserved. Original runs retain large
artifacts and raw audit traces. Benchmark alignment uses upstream evaluators on a
custom task and is not an official ReproduceBench leaderboard score. Generated
code may require license review before redistribution; no new license is assigned.
