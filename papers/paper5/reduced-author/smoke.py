"""Measure parameters and exercise a real full-batch optimizer step per model."""
import json
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, '/workspace')
import torch
import experiment as exp

p = json.load(open('/inputs/protocol.json'))
z, marks, _, _ = exp.data('/inputs/dataset.csv', p)
ds = exp.Windows(z, marks, p, p['lookback'], p['split']['train_end'])
results = []
for name in p['models']:
    exp.seed(p['seed'])
    model = exp.Model(SimpleNamespace(**exp.config(p, name))).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=p['training']['learning_rate'])
    batch = next(iter(exp.DataLoader(ds, batch_size=p['training']['batch_size'])))
    start = time.monotonic()
    prediction = exp.forward(model, batch)
    assert prediction.shape == batch[4].shape
    loss = torch.nn.functional.mse_loss(prediction, batch[4])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    optimizer.step()
    result = dict(model=name, parameters=sum(x.numel() for x in model.parameters()),
                  batch_size=len(batch[4]), loss=loss.item(), seconds=time.monotonic()-start)
    results.append(result)
    print(json.dumps(result), flush=True)
    del model, optimizer, prediction, loss, batch
with open('/workspace/smoke-results.json', 'w') as output:
    json.dump(results, output, indent=2)
