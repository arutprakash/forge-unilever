# Paper6

Author Repo - https://github.com/hqh0728/CrossGNN.git

DataSet - /Users/adelaiderow/forge-unilever/datasets/ETTh1.csv

Paper - https://openreview.net/forum?id=xOzlW2vUYc

```bash
python3 forge.py preflight

python3 forge.py run \
  --paper papers/paper6/paper6.pdf \
  --dataset datasets/ETTh1.csv \
  --author-repo https://github.com/hqh0728/CrossGNN.git \
  --experiment paper6.1 \
  --benchmark-suite autoreproduce
```