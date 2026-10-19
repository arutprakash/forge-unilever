# Paper3

Author Repo - https://github.com/sir-lab/linear-forecasting.git

DataSet - /Users/arut/repo/forge-unilever/datasets/ETTh1.csv

Paper - https://arxiv.org/abs/2403.14587

```bash
python3 forge.py preflight

python3 forge.py run \
  --paper papers/paper3/paper3.pdf \
  --dataset datasets/ETTh1.csv \
  --author-repo https://github.com/sir-lab/linear-forecasting.git \
  --experiment paper3.1 \
  --benchmark-suite autoreproduce
```
