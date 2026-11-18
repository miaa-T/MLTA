# Phase 1 validation record — 18 November 2026

**The standalone notebook passed CPU execution with both training methods. The real
Qwen2.5/MedMCQA benchmark experiment has not been run: this environment reports no CUDA GPU.**

## Checks completed

| Check | Result |
|---|---|
| Notebook schema and code-cell syntax | Passed |
| Fresh-kernel standalone LoRA rehearsal in an isolated working directory | Passed |
| Fresh-kernel standalone full-fine-tuning rehearsal | Passed |
| Multiple-choice scoring and short text generation through lm-evaluation-harness | Passed on synthetic tasks |
| Answer-only token labels, attention/padding masks, nonempty retained dataset | Passed |
| Training, saving and reloading model/adapter | Passed |
| Task scores reconstructed from individual predictions | Matched harness aggregates |
| Tables, loss plot, paired comparison plot, interpretation and audit artifacts | Produced |
| Restart with completed training/evaluation caches | Passed for LoRA |
| Scientific completion remains false for a synthetic rehearsal | Passed for both methods |
| Five paired-statistics unit checks | Passed |
| Real benchmark task registration, including MMLU group and custom PubMedQA | Passed without downloading datasets |
| PubMedQA context rendering | Passed: separate abstract paragraphs use actual newlines |
| Existing `experiments.py list` and fresh synthetic training using `train.py` | Executed successfully |
| Fresh existing-script lower-layer training: upper layers unchanged except save-rounding | Passed; maximum upper-layer difference approximately 0.000030 |

The five statistical checks cover identical predictions, reordered samples, rejection
of changed prompts despite unchanged documents, instruction-weighted IFEval aggregation,
and size-weighted/stratified MMLU aggregation.

The earlier check `tests/check_outputs.py tests/tiny tests/out` failed on the existing
`full_lower` artifacts: supposedly frozen upper layers differed from the current base
by around 0.08–0.12 in maximum absolute weight difference. A fresh run against that same
current base passed, with upper-layer changes around float16 save-rounding scale.
This supports inconsistent/stale saved artifacts, rather than a reproduced layer-selection
bug. The exact history of those old files is unknown. The old complete smoke suite
must not be reported as passing in this session.

## How to reproduce the checks

From the project root, using the project interpreter:

```powershell
.venv/Scripts/python.exe tests/test_phase1_statistics.py
.venv/Scripts/python.exe tools/validate_phase1.py
.venv/Scripts/python.exe tools/validate_phase1.py --method full
```

The first notebook execution produces fresh training and evaluation files. Repeating
the same run exercises completed-result reuse. For a deliberately new experiment,
change `TRIAL` in the notebook; do not delete or mix old results to make checks pass.

Executed copies and their artifacts are stored under:

- `tests/out_phase1/lora/phase1_lab.executed.ipynb`
- `tests/out_phase1/full/phase1_lab.executed.ipynb`

These are local generated artifacts excluded from Git. The source notebook and generator
are included; neither depends on those generated files. The reference environment contains
Python 3.14.2, torch 2.15.0, transformers 5.23.0, peft 0.21.3, datasets 5.2.0,
accelerate 1.15.0 and lm_eval 0.4.13. Each run records its full resolved environment.

## What the rehearsal means

Tiny random models and synthetic questions check execution and numerical plumbing.
They do not establish medical competence, general retention or catastrophic forgetting.
Zero scores and degenerate bootstrap intervals are unsurprising in this rehearsal.
The notebook explicitly labels these figures as software checks and refuses scientific
completion in smoke/quick modes.

## Still required on a GPU

1. Execute `MODE="quick"` to validate real model/dataset downloads and all real scorers.
2. Execute `MODE="research"` with Qwen2.5-1.5B-Instruct and the approved naive LoRA
   substitute, or explicitly choose full fine-tuning with sufficient memory.
3. Confirm the retained training counts, all five benchmark sample counts, saved paired
   predictions and `completion_audit.json`.
4. Interpret real effect sizes and uncertainty, including any finding of no clear forgetting.

Real-data schema compatibility, GPU memory requirements, mixed precision, actual IFEval
generation/scoring, runtime estimates, interrupted-training resume and scientific findings
have not been verified by the CPU rehearsal. Real task registration alone does not verify
dataset downloading or GPU execution. The executed synthetic generation task is not IFEval.
