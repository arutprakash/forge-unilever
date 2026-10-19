# Independent paper-only FITS experiment

Run from /workspace with the supplied inputs mounted under /inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py train > training.log 2>&1
```

This repeats training, validation checkpoint selection, and test evaluation. It overwrites checkpoints/FITS.pt, output/FITS.npy, results.json and training.log. Installed Python, PyTorch, NumPy and pandas suffice; no downloads are needed.

Inference only:

```sh
python run.py predict --checkpoint checkpoints/FITS.pt --dataset /inputs/dataset.csv --protocol /inputs/protocol.json --output /replay-output/FITS.npy
```

See replay.json for the minimal file manifest, and implementation.md for paper mapping and assumptions. Outputs are standardized float32 [window, horizon, feature] arrays. Source imports only local models.py and generic installed packages.
