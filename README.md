# Where Does Forgetting Live?

MLTA 2026 project: Mahdia Toubal, Yousra Meflah, Osman Yasir Ozyilmaz.

We fine-tune a small LLM (Qwen2.5-1.5B-Instruct) on medical questions, measure
how much general skill it loses, test ways to reduce the loss, and look at
where in the model the change happens.

## One-week plan

| Day | Goal | Done when |
|---|---|---|
| 1 | Setup. Quick test on the 0.5B model. Score the base model. | `results/base.json` exists |
| 2 | Naive fine-tuning, then the reference LoRA run. **Check that forgetting is visible.** | general score of `naive` is clearly below `base` |
| 3 | Mitigation runs: rank, learning rate, replay | 3 more result files |
| 4 | Where: attention vs MLP, lower vs upper layers, layer analysis | 4 result files + analysis figures |
| 5 | Model merging, RAG, final recipe | merge and RAG results, `final` run |
| 6 | Pareto plot, demo, figures for the report | `results/pareto.png` |
| 7 | Report and slides | |

Day 2 is the checkpoint. If the naive model does not lose general skill, make
it more aggressive (more epochs, higher learning rate) before going on,
because every later experiment depends on it.

Suggested split: one person runs training and evaluation, one does layer
analysis and merging, one does RAG and the demo. LoRA runs also fit on a free
Colab T4, so teammates can run them in parallel on their own accounts.

## Setup

Colab: put this folder in Google Drive and open `colab.ipynb`.

Laptop (8 GB GPU): `pip install -r requirements.txt`. Use the 0.5B model to
test, or add `--extra "--load_4bit"` to train the 1.5B model.

## Steps

```bash
# 0. test the whole pipeline in a few minutes (0.5B model, tiny data)
python experiments.py run base lora --quick --model Qwen/Qwen2.5-0.5B-Instruct

# 1. base model scores
python experiments.py run base

# 2. fine-tune and score (one or several names)
python experiments.py list
python experiments.py run naive          # full fine-tuning, A100 only
python experiments.py run naive_lora     # use this if you skip the A100
python experiments.py run lora lora_r64 lora_lowlr replay20
python experiments.py run attn_only mlp_only lower_half upper_half

# 3. where did the model change?
python layer_analysis.py --name naive --run runs/naive

# 4. model merging (no training)
python experiments.py merge naive 0.25 0.5 0.75

# 5. RAG
python rag.py build
python rag.py eval --name base
python rag.py eval --name lora --run runs/lora

# 6. table and plot of everything
python pareto.py

# 7. demo
python demo.py lora=runs/lora replay20=runs/replay20 --share
```

Training and evaluation skip work that is already done, so you can rerun a
command after a Colab disconnect.

## Experiments

| Name | What changes | Question it answers |
|---|---|---|
| `base` | no training | starting point |
| `naive` | full fine-tuning | how bad is forgetting? |
| `naive_lora` | aggressive LoRA (rank 64, high learning rate, 3 epochs) | same, without an A100 |
| `lora` | LoRA rank 16 (reference) | does LoRA alone reduce forgetting? |
| `lora_r64` | rank 64 | does rank matter? |
| `lora_lowlr` | learning rate 5e-5 | does a lower learning rate help? |
| `replay20` | 20% general data in the mix | does replay help? |
| `attn_only`, `mlp_only` | LoRA on one block type | attention or MLP? |
| `lower_half`, `upper_half` | LoRA on half the layers | early or late layers? |
| `<run>_merge50` | weights halfway back to base | does merging recover general skill? |
| `<run>+RAG` | passages added to medical questions | can retrieval replace fine-tuning? |

Minimum set if time is short: `base`, `naive` (or `naive_lora`), `lora`,
`replay20`, `attn_only`, `mlp_only`, merging, RAG on `base`.

## Reading the results

- Medical score = mean of MedQA and PubMedQA. General score = mean of MMLU,
  HellaSwag and IFEval (instruction following).
- Test sets are subsets to save compute (see `TASKS` in `evaluate.py`). With
  500 to 1,300 questions the uncertainty is about 1.5 to 2 points, so treat a
  difference under 3 points as noise. `results/<name>.json` has the standard
  error for each score.
- MedMCQA is the training data. MedQA and PubMedQA are different datasets, so
  the test questions were not seen in training.

- IFEval is usually the first score to drop, because the model is trained to
  answer with one letter and then has to write free text again.

## Compute (rough estimates, measure with your first run)

- LoRA run on a T4: about 30 to 45 min training, about 45 min evaluation.
- Whole plan: about 15 to 20 T4 hours, plus under 1 A100 hour for `naive`.
- Full fine-tuning needs about 25 GB of GPU memory, so A100 only.

## What has been tested

`bash tests/smoke_test.sh` passes with torch 2.15, transformers 5.23, peft 0.21,
datasets 5.2 and lm_eval 0.4.13. It runs training (LoRA and full), merging,
layer analysis, the RAG scorer, the Pareto table and `evaluate.py` through
lm-eval, on a tiny random model with fake data, and checks the numbers.

Not tested yet, because they need Hugging Face downloads and a GPU: the real
model and datasets, 4-bit loading, building the textbook index, the demo, and
the time estimates above. The quick test in step 0 covers most of these.

## Files

| File | Role |
|---|---|
| `experiments.py` | list of runs, one command to train and score |
| `train.py` | LoRA or full fine-tuning |
| `evaluate.py` | benchmarks with lm-evaluation-harness |
| `layer_analysis.py` | weight change and hidden-state drift per layer |
| `merge.py` | weight interpolation between base and fine-tuned |
| `rag.py` | textbook index and MedQA with retrieval |
| `pareto.py` | summary table and trade-off plot |
| `demo.py` | side-by-side chat demo |
| `tasks/` | PubMedQA task for lm-eval (parquet copy of the dataset) |
| `tests/` | `bash tests/smoke_test.sh` runs every step on a tiny random model |

## Data

[MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa),
[MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf),
[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA),
[MMLU](https://huggingface.co/datasets/cais/mmlu),
[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag),
[IFEval](https://huggingface.co/datasets/google/IFEval),
[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca),
[MedQA textbooks](https://huggingface.co/datasets/MedRAG/textbooks).
# MLTA
