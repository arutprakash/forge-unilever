# paper3.1 reproduction benchmark

Outcome: numerical_match. Execution rate: 100%.

Reference: fresh author inference replay; published numbers are context only.

| Model | Metric | Author | Generated | Relative gap | Within tolerance |
|---|---|---:|---:|---:|---|
| OLS | mse | 0.373436 | 0.373436 | +0.00% | True |
| OLS | mae | 0.399282 | 0.399282 | +0.00% | True |
| OLS_IN | mse | 0.373384 | 0.373384 | +0.00% | True |
| OLS_IN | mae | 0.399389 | 0.399389 | +0.00% | True |

## Execution and provenance

- author: evaluated; 137.9 seconds. 
- paper-only: evaluated; 137.5 seconds. 

Inspect stage implementation reports, replay execution records, and audit traces for method fidelity or copied-result concerns. Numerical agreement is not proof of architectural correctness.
