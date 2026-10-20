# paper4.1 — preserved experiment

Source run: `20261019T173406Z-569e9b`. Generated source is copied without modification.
See [numerical results](results/RESULTS.md), [protocol](protocol.json), and
[provenance](provenance.json). No alignment evaluation is included.

## Reproduce from the repository root

```bash
python3 forge.py run \
  --paper PATH_TO_ORIGINAL_PDF \
  --dataset PATH_TO_ORIGINAL_CSV \
  --author-repo https://github.com/hqh0728/TimeBase.git \
  --author-ref 369b330f3d77371fcc7e8c75c808d01330c40899 \
  --protocol papers/paper4/paper4.1/committed/20261019T173406Z-569e9b/protocol.json \
  --experiment paper4.1
```

Use the original input files matching the SHA-256 hashes in provenance. Runtime
dependencies and image identities are recorded there; the pipeline Dockerfile
provides the pinned environment. Code entry points are documented in
`generated-code/README.md`. Paths such as `/inputs` refer to container mounts.
Checkpoints are excluded, so inference replay requires retraining or the original
ignored run archive. Exact rerun scores may vary with model generation and training.

## Verify this export

```bash
python3 forge.py verify --run-dir papers/paper4/paper4.1/committed/20261019T173406Z-569e9b
```

Only small text source and reports are preserved. Original runs retain large
artifacts and raw audit traces. Benchmark alignment uses upstream evaluators on a
custom task and is not an official ReproduceBench leaderboard score. Generated
code may require license review before redistribution; no new license is assigned.
