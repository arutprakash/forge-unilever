Repeat both fits, validation selection and test evaluation with:

```sh
python experiment.py train > training.log 2>&1
```

Inputs default to /inputs/dataset.csv and /inputs/protocol.json. Requires installed Python, NumPy, pandas and SciPy. All numerical thread environment variables are set to one before imports. Model-specific fitting: add `--model OLS` or `--model OLS_IN`.

Inference only (no fitting):

```sh
python experiment.py predict --checkpoint checkpoints/OLS.npz --dataset /inputs/dataset.csv --protocol /inputs/protocol.json --output /replay-output/OLS.npy
python experiment.py predict --checkpoint checkpoints/OLS_IN.npz --dataset /inputs/dataset.csv --protocol /inputs/protocol.json --output /replay-output/OLS_IN.npy
```

Predictions are float64 standardized values, [2786,96,7], target starts 11426 through 14211 inclusive. Checkpoints contain coefficients and training-only scaling statistics, no examples or target caches. See implementation.md for derivation and assumptions and metadata/*.json for measured fit/selection results.
