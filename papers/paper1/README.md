# Research Papers

Paper 1 

1	Simple and linear models	DLinear/NLinear	2023	AAAI 2023	Established (2023-24)	Linear	One linear map from lookback to horizon; DLinear splits trend (moving average) and remainder, NLinear subtracts the last value first.	Paper	Code	ETTh1, lookback 336, horizon 96: DLinear MSE 0.375 / MAE 0.399; NLinear MSE 0.374 / MAE 0.394 (paper Table 2)	Confirmed	Exact match. Same table: Linear 0.375/0.397 at H96.	CPU-ok	Easy	High	Apache-2.0	8,150		Extremely widely copied, almost certainly memorised. Original code uses drop_last=True on the test loader (bug flagged by FITS authors), which shifts ETTh1 numbers slightly.		


Author Repo - https://github.com/cure-lab/LTSF-Linear

DataSet - /Users/arut/repo/forge-unilever/datasets/ETTh1.csv

Paper - https://arxiv.org/abs/2205.13504 

