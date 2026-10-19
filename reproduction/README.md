# Centralized paper reproduction pipeline

## Learning-rate schedules

`training.schedule` accepts `"constant"` or a typed delayed exponential decay:

```json
{"type": "exponential", "factor": 0.8, "start_epoch": 4}
```

Epochs start at 1. Before epoch `e`, set LR to
`base_lr * factor**max(0, e-start_epoch+1)`. Thus epochs 1–3 retain the initial
LR and epoch 4 first applies the factor. Both execution sessions are instructed
to apply and log these rates; the metric evaluator does not independently prove
training-schedule adherence. Other schedules and prose descriptions are rejected
with a schedule-specific error. Existing constant protocols remain supported.

## Git-trackable exports

Successful `run` and `benchmark` commands automatically create immutable curated
exports in `<experiment>/committed/<run-id>/` (or `<run-id>--<evaluation-id>/`
when an evaluation is included). Use `--no-export` to disable this step. Export
failure exits nonzero; original sealed results remain available. A suite failure
still exports the completed numerical experiment before reporting the error.

Export an existing completed run and its matching evaluation without Docker:

```bash
python3 forge.py export --run-dir PATH_TO_RUN --evaluation-dir PATH_TO_EVALUATION
```

The evaluation flag is optional. Exports verify source seals first and preserve
small generated source, protocol, author/generated metrics, numerical reports,
optional benchmark scores/explanations, input hashes, author commit, image IDs,
and original seal references. Each export has its own seal and reproduction
instructions. Existing exports are verified and never overwritten. Partial runs
or mismatched evaluations are rejected. Author checkouts, credentials, raw
transcripts, input PDFs/CSVs, checkpoints, prediction arrays, and caches remain in
the ignored archives. Exports do not contain a complete inference replay archive.

`committed/` is Git-trackable; creating an export does not itself stage, commit,
or push anything. Review and commit it explicitly with the pipeline source changes.

Run from the repository root with just the three inputs:

```sh
python3 forge.py run \
  --paper papers/paper1/paper1.pdf \
  --dataset datasets/ETTh1.csv \
  --author-repo https://github.com/cure-lab/LTSF-Linear
```

The command builds missing Docker images, selects or designs a neutral protocol,
clones a fresh author checkout, runs the author implementation, starts a fresh
paper-only Codex session to generate and train new code, independently replays
inference from saved source/checkpoints, and benchmarks both implementations.
It uses your existing **Codex ChatGPT login**, not an API key. Source, logs,
checkpoints, initial predictions, replayed predictions, metrics, and a report
are all retained. Existing experiments/results are never used as fallback data.

Requirements: running Docker Engine/Desktop, a non-root host user, Python 3.9+,
and a file-backed Codex login (`~/.codex/auth.json` or `$CODEX_HOME/auth.json`).
No scientific Python dependencies are required on the host to operate the CLI;
they are version-pinned in the runtime image. Linux/amd64 CPU images are built
(Docker Desktop emulates them on Apple Silicon). Allocate at least 4 GB to Docker.

```sh
python3 forge.py preflight
python3 forge.py plan --paper PATH.pdf --dataset PATH.csv --author-repo https://github.com/OWNER/REPO
```

Use `--experiment paper1.3` to put new runs under
`papers/paper1/paper1.3/runs/<run-id>/`. Without it, the default is
`<PDF parent>/runs/<run-id>/`. For a PDF outside the repository, its default
paper home is `<repo>/papers/<PDF stem>/`, so outputs still stay in the repo.
`--output-root PATH` overrides the run parent. `--run-id NAME` gives a stable
name but refuses to overwrite an existing run.
Inputs can be outside the repo. Add `--author-ref COMMIT_OR_TAG` for exact source
replay, `--model MODEL_NAME` to pin an account-available Codex model, and
`--build` to force an image rebuild. Otherwise the pinned CLI's default model is
used. Full legacy configs remain supported with `--config PATH.json`.

## What is supported

The current adapter supports **regular time-series forecasting CSVs**, numeric
feature columns, chronological splits, training-only standardization, and
multivariate standardized MSE/MAE. It is not a universal adapter for every paper
in the workbook: classification, causal discovery, irregular/event data,
pretrained-model downloads, and GPU workloads need separate adapters/images.

For the DLinear repository, the default preset is ETTh1, lookback 336, horizon
96, Linear/NLinear/DLinear, selected from `Core 30!C2:P2` in the supplied workbook.
The workbook rates it CPU-ok/easy/high-specification, with suitability 5/5 in
`All candidates!AE2`. This is the initial inexpensive anchor, with the workbook's
memorization/partial-test-batch caveat retained in preset provenance.

For an unfamiliar repository, an additional **fresh PDF-only design session**
reads only the paper and CSV, proposes a forecasting protocol, and documents its
assumptions. The host validates its schema, feature names, timestamps, sample
counts, and split sizes before freezing it. This happens **before the author
repository is fetched**. No author URL, reported performance, code, or free-form
design notes enter the neutral protocol supplied to the generated-code session.
Unsupported tasks fail explicitly. To fix the protocol yourself, use
`--protocol PATH.json`; the preset's `protocol` object is an example schema.

The initial preset uses train/validation/test row boundaries 8640/11520/14400,
seed 42, Adam LR 0.001, batch 64, at most 50 epochs, patience 5, and all 2,785
test windows. These are shared experimental controls, not a claim to recover
the original paper's full training recipe. Different architectural interpretations
can remain; both implementation reports must document them.

## Workbook-backed benchmarking

The host-only `benchmark-policy.json` records the source workbook SHA-256 and
exact sheet/cell references. It is never mounted in an agent's worker.
This is a transparent **Forge scoring policy inspired by the workbook**, not an
official implementation or a claim to reproduce those benchmark suites:

| Workbook entry | Applied here |
|---|---|
| AutoReproduce, `Related benchmarks!A10:K10` | Execution rate and relative gap against a freshly executed author reference |
| RECLAIM, `A4:K4` | Pin protocol/metric/tolerance before execution; trace reported numbers to run artifacts |
| AutoExperiment, `A9:K9` | Explicit 5% absolute relative-difference band for each model's MSE and MAE |
| Artisan, `A2:K2` | Distinguish execution failure from numerical mismatch; retain copying/method-fidelity review as a separate question |

Relative gap is `(generated - author) / abs(author)`; positive means worse for
MSE/MAE. Every model and both metrics must fall within the predeclared 5% band
for `numerical_match`. Large improvements are `metric_mismatch` too, because
numerical reproduction differs from outperforming a reference. With a zero
reference, only exact zero matches; other relative gaps are null rather than
infinite. The execution rate is the fraction of the **two implementation stages**
that completed all requested models, fresh inference replay, and independent
evaluation (0, 0.5, or 1). It is not a multi-attempt Pass@k estimate.

Published workbook values are kept as context only. They do not determine
pass/fail because shared training controls may differ from published recipes.
Tokens and model identifiers are taken from actual CLI events where available;
missing data remains null. Dollar costs are not guessed from subscription usage.
Wall times and per-command child CPU/RSS information are logged separately;
worker resource counters are reported measurements, not verified model-only peaks.

Initial prediction arrays alone cannot complete a stage. Each agent must declare
minimal source/checkpoints and Python inference commands in `replay.json`. A new
offline container receives those files, the PDF/CSV/protocol, and an empty output
directory. It receives **no cached predictions and no author checkout**. A separate
trusted evaluator reads only numeric `.npy` files, recomputes all metrics, and
checks that fresh inference matches the initial exported predictions.

Numerical agreement and replay demonstrate execution evidence, not full semantic
fidelity, correct checkpoint use, absence of target leakage, or absence of
memorized/hard-coded code. The report explicitly leaves those assessments open
for source/trace review. It does not automatically accuse a run of copying or
invent an LLM alignment score.

## Folder layout and audit trail

```text
<run parent>/<run-id>/
  manifest.json                     # state, exact inputs/config, author SHA, image IDs
  benchmark-policy.json             # policy frozen before any agent executes
  benchmark.json                    # outcome, execution rate, per-model gaps, timings
  comparison.json
  RESULTS.md
  audit-seal.json                   # SHA-256 inventory of all regular artifacts
  audit/
    events.jsonl                    # timestamped, hash-linked lifecycle/command events
    commands/                       # numbered stdout/stderr logs for Docker operations
    failure.log                     # on errors, with traceback
  inputs/                           # only PDF, CSV, frozen protocol
  protocol-design/                  # optional isolated planning session and its audit
  author/
    repo/                           # freshly fetched author checkout
    run/workspace/                  # wrapper/patches, source, checkpoints, initial output
    run/replay-source/              # declared inference source/checkpoint snapshot
    run/replay/output/              # newly executed predictions and inference logs
    run/results/                    # trusted metrics.json and summary.csv
    run/audit/                      # prompts, transcripts, inspections, command logs
      controller/tool-calls.jsonl   # commands/responses captured outside worker access
  paper-only/
    generated-code/                 # fresh source, implementation.md, checkpoints, output/
    replay-source/
    replay/output/
    results/
    audit/
```

Run lifecycle events include UTC times, sequence numbers, previous-record hashes,
command argv, exit codes, durations, and output-file hashes. Input prompt hashes
link to separately retained prompts/transcripts. The generated worker never
mounts the host audit or controller's command log. Model-issued commands and
their responses are recorded by the bridge outside that worker. Docker build
logs are included for first-run builds, including dependency installation errors.
Standalone `build` logs go under `reproduction/builds/`.

```sh
python3 forge.py verify --run-dir PATH_TO_RUN
```

Verification detects altered/missing regular files, added unsealed files, and
changed journal records. It checks integrity, not scientific success. The seal is
not a digital signature: archive its printed SHA-256 outside the run directory
to detect replacement of the entire run/seal by someone with host write access.
Input URLs and file paths are logged; plaintext login tokens are not. A private
credentials-only Docker volume carries token refreshes between fresh controllers
and is deleted at experiment end. Host login files are read-only and never
copied into an image, prompt, worker, or run artifact.

An author failure does not expose its messages to the paper-only agent or stop
an independent generated-code attempt. Partial work and errors remain available;
a comparison cannot pass without both stages. Infrastructure failures also
produce a failed manifest, benchmark outcome, journal, and seal. No old experiment
metrics are substituted. A completed workflow can legitimately report
`metric_mismatch`; check benchmark outcome, not just process exit status.

## Docker isolation

All model-issued commands run in workers with `--network none`, non-root UID,
read-only root, capabilities dropped, no-new-privileges, and fixed memory/CPU/PID
limits. Only the author worker mounts the author repository, read-only. Paper-only
and protocol-design workers receive only their clean inputs/workspace/IPC.
No worker gets a Docker socket, host home, Codex credentials, prior runs, or audit.

Fresh Codex controllers have no shell/web tools, inherited skills, apps, hooks,
plugins, memories, or earlier-session history. They use a stage-local Unix-socket
MCP bridge to the offline worker. Controllers are on internal Docker networks;
an exact-host CONNECT proxy permits only OpenAI inference/authentication hosts.
Remote inference needs that connection; model execution has no internet access.
The proxy tunnels TLS rather than inspecting its contents, so disabling local
controller tools is part of the trust boundary. This prevents runtime source
retrieval, not pretrained memorization or kernel/container-runtime exploits.

## Validation

```sh
python3 -m unittest discover -s reproduction/tests -v
RUN_DOCKER_TESTS=1 python3 -m unittest discover -s reproduction/tests -v
```

NumPy is needed on the host for evaluator tests; the rest use the standard
library. Docker integration is opt-in and requires built images. Docker is not
available on the development machine, so real image builds, container isolation,
Codex-in-container login, and full model runs remain unverified. The earlier
paper1.2 command now forwards to this central implementation.

Local checks cover neutral protocol validation, design-before-fetch ordering,
author/generated isolation, replay snapshot restrictions and prediction matching,
benchmark tolerance boundaries, missing/invalid outputs, failure reporting, audit
chain tampering, file seals, and backward-compatible invocation. Live Docker
execution is a separate verification step and is not implied by those unit tests.
# Upstream benchmark evaluation

Run the pinned AutoReproduce evaluators automatically after a complete experiment:

```bash
python3 forge.py run --paper papers/paper2/paper2.pdf --dataset datasets/ETTh1.csv --author-repo https://github.com/VEWOXIC/FITS --experiment paper2.1 --benchmark-suite autoreproduce
```

Evaluate an existing sealed run without training again:

```bash
python3 forge.py benchmark --run-dir papers/paper2/paper2.1/runs/20261017T050224Z-70390f
```

Optional `--model` selects the judge available to the existing Codex login. No API key is required. Outputs are stored outside the source run under `evaluations/<source-run>/<evaluation-id>/`; verify that folder with `forge.py verify --run-dir PATH`. Evaluation failure exits nonzero and retains a sealed partial report. The original experiment seal is checked before and after evaluation.

The integration downloads [AutoReproduce](https://github.com/AI9Stars/AutoReproduce) at commit `e1a6c9220fadd9f6fae3b5e891eda7fe18989223`. Its unchanged `summarize_points.py`, `eval_high.py`, `eval_low.py`, `eval_mixed.py`, and `cal_per_gap.py` functions construct evaluation prompts and compute performance gaps. An offline adapter replaces `query_model` with prompt capture; separate tool-free Codex sessions judge the captured prompts using the existing login. Sources, prompts, responses, hashes, Docker commands, judge usage, and adaptations are retained. Recursive implementation source is bundled as text because upstream only reads top-level Python files. Candidate/reference source is never executed by this evaluation adapter.

Reported scores: paper/high alignment (five × 20), code/low alignment (four × 25), mixed alignment (five × 20), execution rate, and each model/metric's upstream gap `abs(reference-generated)/max(reference,generated)`. Component scores are validated and summed locally; missing or ambiguous scores fail rather than being invented. These gaps are separate from Forge's author-denominator 5% numerical agreement policy.

This is **an upstream-evaluator integration on a custom task**, not the official 13-task ReproduceBench suite or a comparable leaderboard score. FITS uses Forge's frozen protocol and executed reference rather than a curated ReproduceBench task. The judge transport/model differs from upstream's default `o1`/temperature 0.6; the wrapper treats code/paper text as untrusted. Retrospective key points are not preregistered, and model judgments do not prove correctness or absence of copying. No benchmark code, reference code, or evaluation output enters the independent generation stage. To evaluate the official task collection, its curated datasets, task environments, references, and prescribed judge settings would be needed separately.
