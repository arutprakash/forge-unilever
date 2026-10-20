# paper6.1 reproduction benchmark

Outcome: metric_mismatch. Execution rate: 100%.

Reference: fresh author inference replay; published numbers are context only.

| Model | Metric | Author | Generated | Relative gap | Within tolerance |
|---|---|---:|---:|---:|---|
| CrossGNN | mse | 0.450410 | 0.917804 | +103.77% | False |
| CrossGNN | mae | 0.449944 | 0.680099 | +51.15% | False |

## Execution and provenance

- author: evaluated; 412.0 seconds. 
- paper-only: evaluated; 157.9 seconds. 

Inspect stage implementation reports, replay execution records, and audit traces for method fidelity or copied-result concerns. Numerical agreement is not proof of architectural correctness.
