# paper2.1 reproduction benchmark

Outcome: numerical_match. Execution rate: 100%.

Reference: fresh author inference replay; published numbers are context only.

| Model | Metric | Author | Generated | Relative gap | Within tolerance |
|---|---|---:|---:|---:|---|
| FITS | mse | 0.438461 | 0.458161 | +4.49% | True |
| FITS | mae | 0.446152 | 0.462717 | +3.71% | True |

## Execution and provenance

- author: evaluated; 119.1 seconds. 
- paper-only: evaluated; 141.7 seconds. 

Inspect stage implementation reports, replay execution records, and audit traces for method fidelity or copied-result concerns. Numerical agreement is not proof of architectural correctness.
