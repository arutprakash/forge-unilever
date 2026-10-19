# Independent TimeBase experiment

Repeat training, validation checkpoint selection, and test evaluation from /workspace:

```sh
python run.py > training.log 2>&1
```

Inference only (loads the saved checkpoint; does not train):

```sh
python run.py --mode predict --checkpoint checkpoints/TimeBase.pt --dataset /inputs/dataset.csv --protocol /inputs/protocol.json --output /replay-output/TimeBase.npy
```

Requires installed Python, PyTorch, NumPy, and pandas. No installation or network access is needed. `replay.json` lists the complete inference file set. Outputs are standardized float32 arrays with axes [window, horizon, feature]. See implementation.md for paper mapping, assumptions, and run results.
