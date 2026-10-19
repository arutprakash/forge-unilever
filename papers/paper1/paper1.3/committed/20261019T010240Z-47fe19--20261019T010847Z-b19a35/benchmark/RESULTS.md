# AutoReproduce custom-task evaluation

Status: complete.

custom-task evaluation; not an official ReproduceBench leaderboard result

| Alignment evaluator | Score / 100 |
|---|---:|
| high | 100 |
| low | 94 |
| mixed | 82 |

Execution rate: 100%.

| Model | Metric | Upstream performance gap |
|---|---|---:|
| Linear | mse | 1.92% |
| Linear | mae | 1.56% |
| NLinear | mse | 0.68% |
| NLinear | mae | 0.83% |
| DLinear | mse | 0.39% |
| DLinear | mae | 0.44% |

## Adaptations

- Existing Codex ChatGPT login replaces upstream query_model API transport; upstream o1/temp=0.6 is not reproduced.
- Recursive Python source is bundled into one top-level file for upstream glob readers; no candidate/reference code executes.
- FITS and the frozen Forge protocol replace curated ReproduceBench tasks/reference code.
- Key points are generated retrospectively from PDF and frozen protocol, not preregistered for this existing run.
- Source and paper text are untrusted judge inputs; alignment scores are model judgments, not proof of correctness.
