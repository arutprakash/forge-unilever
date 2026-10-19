# Paper2

Author Repo - https://github.com/VEWOXIC/FITS.git

DataSet - /Users/arut/repo/forge-unilever/datasets/ETTh1.csv

Paper - https://arxiv.org/abs/2307.03756


```bash
python3 forge.py preflight

python3 forge.py run \
  --paper papers/paper2/paper2.pdf \
  --dataset datasets/ETTh1.csv \
  --author-repo https://github.com/VEWOXIC/FITS.git \
  --experiment paper2.1 \
  --benchmark-suite autoreproduce
```