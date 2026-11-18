# Phase 1: guided execution and implementation review

Read this first, then open [phase1_lab.ipynb](phase1_lab.ipynb). The notebook is a
self-contained practical lab: it imports no project modules, requires no previous
notebook state, and contains its own preparation, training, evaluation and reporting.
The default CPU rehearsal deliberately cannot claim scientific completion.

## What the project is trying to establish

The proposal asks whether medical specialization improves a small language model's
medical performance at the cost of general knowledge, commonsense reasoning, or
instruction following. Later phases ask where changes occur and how to reduce losses.

The roadmap's Phase 1 deliverable is the original model versus one naively trained
model, including medical and general scores and an assessment of forgetting. It does
not require forgetting to be observed. The proposal makes full fine-tuning optional,
while the roadmap places LoRA in Phase 2. You approved a **clearly labeled naive LoRA
substitute when full training will not fit**. The lab supports both and records which
method was used. Results from these methods answer different experimental questions.

Phase 1 includes:

1. Set up the environment and load Qwen2.5-1.5B-Instruct.
2. Prepare medical-only MedMCQA training data.
3. Prepare MedQA, PubMedQA, MMLU, HellaSwag and IFEval evaluation.
4. Evaluate the original model before training.
5. Train one naive model with declared settings.
6. Evaluate the trained model on the same questions under the same protocol.
7. Compare scores, uncertainty and practical effect sizes; document the findings.

Layer-wise analysis, replay, learning-rate/rank sweeps, RAG, merging, a Pareto search
across mitigation methods and a deployment demo are outside this notebook's scope.

## Map of the existing implementation

| Existing file | Main entry points and role | Phase 1 relevance |
|---|---|---|
| `experiments.py` | `RUNS`, `main`: selects settings and launches training/evaluation subprocesses | Orchestration; `naive` is full training, `naive_lora` is aggressive LoRA |
| `common.py` | `medical_examples`, `mcq_prompt`, `chat`: load, select and format examples; `load_model`: base/full/adapter loading | Training data and shared model helpers |
| `train.py` | `encode`: mask prompt loss; `Collator`: pad a batch; `build_data`, `build_model`, `main`: train and save | Main training implementation |
| `evaluate.py` | `TASKS`, `main`: load an HFLM wrapper and call `simple_evaluate` | Required benchmark scoring |
| `tasks/pubmedqa_parquet.yaml` | Defines PubMedQA prompt, answers, split and metric | Required custom benchmark |
| `pareto.py` | Reads complete JSON files, computes medical/general means and plots | Can summarize Phase 1, but also mixes later-phase runs |
| `colab.ipynb` | Drive mounting and shell commands | Covers all phases; not an independent Phase 1 lesson |
| `layer_analysis.py`, `merge.py`, `rag.py`, `demo.py` | Weight/activation changes, interpolation, retrieval and UI | Later phases |
| `tests/` | Tiny random model and synthetic pipeline checks | Software validation only |

The execution flow is `experiments → train/common → saved checkpoint → evaluate → JSON → summary`.
The original notebook is a launcher, not the implementation itself.

`train.encode` builds the prompt and full chat, places `-100` on prompt positions,
and keeps answer token IDs as labels. `Collator` pads inputs, masks padding in
attention, and gives padding labels `-100`. Thus the question provides context but
only the assistant answer contributes to cross-entropy. Overlength examples are dropped.

`build_model` either enables all parameters (full training) or adds LoRA matrices to
selected projection modules. `Trainer` performs optimization; `save_pretrained`
writes a full checkpoint or just the adapter. Evaluation reloads it and measures
held-out behavior. Training loss and benchmark accuracy are different quantities.

## Findings from the review

The core training/evaluation architecture already exists. The missing piece was a
Phase 1-only teaching notebook with trustworthy provenance and interpretation.

| Finding | Consequence / treatment in the new lab |
|---|---|
| Original evaluation skips tasks based only on their name in a JSON file | Different models/limits can inherit old scores; new lab fingerprints configuration and checkpoint weights |
| Original training skips when `run_config.json` exists, without matching arguments | New lab checks configuration/data identity and saved weights |
| Original evaluation has `log_samples=False` | Cannot verify paired questions or compute paired uncertainty; new lab saves predictions, document/prompt/target hashes |
| Original README says to increase training until forgetting appears | This selects on test outcomes; new lab predeclares settings and permits inconclusive/negative results |
| Original notebook mixes all phases | New lab contains only the Phase 1 workflow |
| Original training has no intermediate checkpoint saving | New lab saves epoch checkpoints and resumes training |
| Most dependencies are unpinned | New lab records resolved versions and `pip freeze`; its harness version is fixed |
| Medical datasets have different names | This alone does not establish absence of duplicate/pretraining contamination; the lab states the limitation |
| README's blanket “under three points is noise” rule | Replaced by paired sample comparisons and exploratory bootstrap intervals |
| Existing test outputs are synthetic | Kept distinct from real medical/general results |

The lab retains prompt formatting and the draft's naive LoRA hyperparameters. It makes
zero-shot evaluation explicit, uses smaller microbatches for memory, filters training
duplicates, records dropped examples, and preserves a clear method label. Therefore
compare the two checkpoints produced by this notebook, not an arbitrary old score file.

## First session: run on this machine

The inspected environment uses Python 3.14.2 and has the ML libraries installed.
PyTorch reported no CUDA device. Jupyter execution packages were installed into the
project virtual environment for notebook validation. `results/` held no real benchmark
results at inspection time. The open `tests/out/eval_results/full.json` contains toy
tasks `t_mc`, `t_group`, `t_pub`, `t_gen` from a random model, not the five real benchmarks.

In a PowerShell terminal at the repository root:

```powershell
.venv/Scripts/python.exe -c "import sys, torch; print(sys.executable); print(torch.cuda.is_available())"
.venv/Scripts/python.exe experiments.py list
```

The first command checks which interpreter is used and whether GPU acceleration is
available. The second lists experiment configurations without training anything.

Open `phase1_lab.ipynb` in VS Code and select `.venv/Scripts/python.exe` as its kernel.
Keep `MODE = "smoke"` and `METHOD = "lora"`. Run each section in order, read its output,
and answer its short interpretation questions. It creates its own tiny model and synthetic
data; downloads are unnecessary after dependencies are installed.

For an automated fresh-kernel rehearsal:

```powershell
.venv/Scripts/python.exe tools/validate_phase1.py
```

That command runs the notebook in `tests/out_phase1/lora/`, away from the repository
modules, and writes an executed notebook there. It checks that research completion
remains false. Add `--method full` to check the full-training code path on a tiny model.

If setting up a new environment, install a PyTorch build suitable for the runtime,
then install `requirements-phase1.txt` using that environment's Python. The notebook
also includes an optional missing-package installation cell. Restart after installation.
See the [official PyTorch setup instructions](https://pytorch.org/get-started/locally/).

## Running the existing scripts deliberately

The following exercises use the existing `tests/tiny` checkpoint and create separate
outputs. The model is random: do not interpret its numbers as medical performance.

```powershell
.venv/Scripts/python.exe train.py --model tests/tiny --synthetic --n_train 64 --epochs 1 --batch_size 8 --grad_accum 1 --method full --lr 0.003 --out tests/out_phase1/draft_full_exercise
```

Read the example-count message, training loss and the saved `run_config.json`.
The model/tokenizer files are needed to reload the trained checkpoint. With fewer
than 20 optimizer updates the original script may not print intermediate losses,
because its logging interval is 20; the final summary still contains training loss.

On a suitable GPU, the draft's real-data quick command is:

```powershell
.venv/Scripts/python.exe experiments.py run base lora --quick --model Qwen/Qwen2.5-0.5B-Instruct
```

This downloads models/data and runs real benchmarks. It is an integration check, not
the 1.5B deliverable. The original orchestrator uses fixed `quick_` output names, so
inspect previous configurations before reusing them. Prefer the new notebook's
isolated outputs for the scientific experiment.

## Second session: real Phase 1 experiment

Move the notebook to a CUDA GPU environment, or open it in Colab with a GPU runtime.
No repository clone is necessary. If using Drive, mount it and set `OUTPUT_ROOT` to
a persistent folder before running the experiment. There is no automatic cloud job submission.

1. Run `MODE = "quick"` to exercise real downloads and all five benchmark integrations.
2. Restart the kernel. Set `MODE = "research"`, `METHOD = "lora"`, and review the declared settings.
3. Execute baseline evaluation, training and tuned evaluation in order.
4. Inspect the saved paired benchmark table, plots, interpretation and completion audit.
5. Retain the entire run directory with the model, configurations, predictions and environment.

The default research LoRA substitute uses 10,000 selected medical examples (before
length filtering), rank 64, learning rate 5e-4, three epochs, no replay and all projection
types. Full fine-tuning uses learning rate 2e-5 and two epochs. Benchmark limits match
the draft, with zero-shot prompting declared explicitly. Actual retained counts and
evaluated sample counts are saved, not assumed.

A score changing from 40% to 44% is **+4 percentage points**, not +4% relative. Positive
medical change and negative general change together support the specialization/retention
trade-off. An interval spanning zero makes the direction inconclusive at that precision.
No clear decline is also a valid finding. Discuss magnitude, task differences, sample
limits, one training seed and possible answer-format effects before calling a loss catastrophic.

## Artifact map

| Artifact inside a run directory | Purpose |
|---|---|
| `configuration.json`, `environment.json`, `requirements-resolved.txt` | Reproduction settings and package versions |
| `selected_training_rows.json`, `training_data_audit.json` | Exact selection, retained IDs and dropped count |
| `evaluation_protocol.json`, `baseline/`, `tuned/` | Benchmark settings, scores, actual questions and predictions |
| `naive_model/`, `training_checkpoints/` | Final model/adapter and resumable training state |
| `training_log.json`, `training_loss.png` | Optimization diagnostics |
| `benchmark_comparison.csv`, `category_summary.csv` | Paired task changes and descriptive category means |
| `phase1_comparison.png`, `interpretation.json` | Visual and written results |
| `completion_audit.json` | Coverage check; research completion must be false for smoke/quick |

The generator `tools/build_phase1_notebook.py` is only a maintenance aid; executing
the notebook does not depend on it. The original implementation is preserved.

Primary technical references: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[MedMCQA dataset](https://huggingface.co/datasets/openlifescienceai/medmcqa),
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
