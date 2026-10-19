# Pipeline fix log

This log records failures, their causes, implementation changes, and validation.
Original run artifacts remain unchanged; a code fix does not imply that an
affected experiment has completed successfully.

## FIX-001 — Support the TimeBase delayed learning-rate decay

Date: 2026-10-19. Status: implemented and validated; full experiment rerun pending
at the time of this entry.

### Failure and cause

Paper 4.1 run `20261019T172728Z-1a1346` stopped during PDF-only protocol design
validation with `Unsupported schedule or checkpoint selection`. Author fetching,
model generation, and training had not started.

The design session selected the paper's delayed decay, written as free text:
`multiply learning rate by 0.8 each epoch after the first three epochs`.
The validator accepted only `"constant"`. The design prompt encouraged
paper-specified training controls without explicitly stating that restriction.
Checkpoint selection (`lowest validation MSE`) was valid; the combined error
message obscured which field failed.

### Changes

- [Protocol validator](reproduction/configuration.py): accepts `"constant"` or a
  strict object `{"type":"exponential","factor":0.8,"start_epoch":4}`. Rejects
  unknown fields, unsupported types, nonfinite/out-of-range factors, and invalid
  starting epochs. Reports schedule and checkpoint errors separately.
- [Design instructions](reproduction/prompts/design.md): specify accepted schedule
  representations and require unsupported choices to be documented.
- [Shared execution instructions](reproduction/prompts/common.md): require both
  implementations to set and log each epoch's learning rate using the same rule.
- [Schedule tests](reproduction/tests/test_schedules.py): cover existing constant
  protocols, structured decay, invalid inputs, and distinct checkpoint errors.
- [Pipeline documentation](reproduction/README.md): documents the schedule contract
  and its verification limits.

Epochs are numbered from 1. Before training epoch `e`, set every optimizer
parameter group's learning rate to:

```text
base_lr * factor ** max(0, e - start_epoch + 1)
```

For the recovered paper 4 protocol:

| Epoch | Learning rate |
|---|---:|
| 1–3 | 0.01 |
| 4 | 0.008 |
| 5 | 0.0064 |

### Recovery artifacts and command

The [corrected protocol](papers/paper4/paper4.1/protocol.json) preserves the
original design choices, changing only the schedule representation. Its
[rationale](papers/paper4/paper4.1/protocol.md) records the source failed run and
the recovery. The original failed run and its audit seal were not modified.

From the repository root:

```bash
python3 forge.py run \
  --paper papers/paper4/paper4.pdf \
  --dataset datasets/ETTh1.csv \
  --author-repo https://github.com/hqh0728/TimeBase.git \
  --experiment paper4.1 \
  --protocol papers/paper4/paper4.1/protocol.json \
  --benchmark-suite autoreproduce
```

This starts a new run using the corrected protocol, skipping protocol generation.

### Validation and limits

- Corrected protocol passed schema and ETTh1 dataset validation.
- `forge.py plan` accepted the recovered configuration with protocol design disabled.
- Test suite: 37 tests run; 36 passed, one optional Docker integration test skipped.
- Whitespace checks passed for the changed implementation and documentation.

The full paper 4 training/evaluation pipeline was not rerun as part of this fix.
Both execution sessions are instructed to implement and log the schedule; the
metric evaluator does not independently establish training-schedule adherence.

For future entries, record a unique fix ID, date, affected run, failure, root cause,
changed files, recovery steps, validation, and any remaining limitations.
