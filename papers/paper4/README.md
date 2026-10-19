#Paper4

Paper - https://proceedings.mlr.press/v267/huang25az.html

Dataset - /Users/arut/repo/forge-unilever/datasets/ETTh1.csv

Author Repo - https://github.com/hqh0728/TimeBase.git



```bash
python3 forge.py preflight

python3 forge.py run \
  --paper papers/paper4/paper4.pdf \
  --dataset datasets/ETTh1.csv \
  --author-repo https://github.com/hqh0728/TimeBase.git \
  --experiment paper4.1 \
  --benchmark-suite autoreproduce
```