# Independent paper-only CrossGNN experiment

Repeat training, validation checkpoint selection, and test evaluation from /workspace:

```sh
python experiment.py > training.log 2>&1
```

Defaults read /inputs/dataset.csv and /inputs/protocol.json. Training saves
best_CrossGNN.pt, training_metadata.json, and output/CrossGNN.npy. All computations
use one CPU thread. No downloads or model packages are used.

Inference only (no optimization):

```sh
python experiment.py --predict --checkpoint best_CrossGNN.pt --dataset /inputs/dataset.csv --protocol /inputs/protocol.json --output /workspace/replayed/CrossGNN.npy
```

replay.json lists the only three files needed for inference and the pipeline's
/replay-output destination. That destination must be writable in the replay
container. The local verification used /workspace/replaycheck/output because the
worker forbids creating directories at its root. Predictions matched bitwise.
See implementation.md for architecture assumptions and measured results.
