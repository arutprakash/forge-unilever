# paper2.1 — preserved experiment

Source run: `20261017T050224Z-70390f`. Generated source is copied without modification.
See [numerical results](results/RESULTS.md), [protocol](protocol.json), and
[provenance](provenance.json). See [alignment evaluation](benchmark/RESULTS.md).

## Reproduce from the repository root

```bash
python3 forge.py run \
  --paper PATH_TO_ORIGINAL_PDF \
  --dataset PATH_TO_ORIGINAL_CSV \
  --author-repo https://github.com/VEWOXIC/FITS \
  --author-ref d040bb015b6299da26d879b90dd19c80fb72c160 \
  --protocol papers/paper2/paper2.1/committed/20261017T050224Z-70390f--20261017T052238Z-d4af6f/protocol.json \
  --experiment paper2.1 \
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
python3 forge.py verify --run-dir papers/paper2/paper2.1/committed/20261017T050224Z-70390f--20261017T052238Z-d4af6f
```

Only small text source and reports are preserved. Original runs retain large
artifacts and raw audit traces. Benchmark alignment uses upstream evaluators on a
custom task and is not an official ReproduceBench leaderboard score. Generated
code may require license review before redistribution; no new license is assigned.
