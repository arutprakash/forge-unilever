# Independent paper-only LTSF-Linear experiment

From /workspace run `python run.py` to repeat all training, validation checkpoint selection, and test evaluation. Inputs default to /inputs/dataset.csv and /inputs/protocol.json. Logs are Linear.log, NLinear.log, and DLinear.log. Best checkpoints are in checkpoints/; standardized float32 predictions and measured metadata are in output/.

Inference only, for example:
`python experiment.py predict --checkpoint checkpoints/Linear.pt --dataset /inputs/dataset.csv --protocol /inputs/protocol.json --output /replay-output/Linear.npy`

replay.json lists the minimal inference files and a command per model. No training occurs in predict mode. Requires installed Python, PyTorch, NumPy and pandas; no model packages or downloads. See implementation.md for paper mapping and assumptions.
