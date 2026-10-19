# paper1.3 reproduction benchmark

Outcome: numerical_match. Execution rate: 100%.

Reference: fresh author inference replay; published numbers are context only.

| Model | Metric | Author | Generated | Relative gap | Within tolerance |
|---|---|---:|---:|---:|---|
| Linear | mse | 0.385807 | 0.378382 | -1.92% | True |
| Linear | mae | 0.408313 | 0.401950 | -1.56% | True |
| NLinear | mse | 0.378630 | 0.381207 | +0.68% | True |
| NLinear | mae | 0.398457 | 0.401790 | +0.84% | True |
| DLinear | mse | 0.381675 | 0.383171 | +0.39% | True |
| DLinear | mae | 0.406028 | 0.407814 | +0.44% | True |

## Execution and provenance

- author: evaluated; 202.7 seconds. 
- paper-only: evaluated; 154.4 seconds. 

Inspect stage implementation reports, replay execution records, and audit traces for method fidelity or copied-result concerns. Numerical agreement is not proof of architectural correctness.
