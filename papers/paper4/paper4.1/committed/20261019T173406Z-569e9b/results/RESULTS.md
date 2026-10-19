# paper4.1 reproduction benchmark

Outcome: numerical_match. Execution rate: 100%.

Reference: fresh author inference replay; published numbers are context only.

| Model | Metric | Author | Generated | Relative gap | Within tolerance |
|---|---|---:|---:|---:|---|
| TimeBase | mse | 0.438663 | 0.441949 | +0.75% | True |
| TimeBase | mae | 0.445181 | 0.446823 | +0.37% | True |

## Execution and provenance

- author: evaluated; 134.1 seconds. 
- paper-only: evaluated; 138.8 seconds. 

Inspect stage implementation reports, replay execution records, and audit traces for method fidelity or copied-result concerns. Numerical agreement is not proof of architectural correctness.
