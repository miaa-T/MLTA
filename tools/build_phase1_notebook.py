"""Build the self-contained teaching notebook; no runtime dependency on this file."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def add(kind, source):
    cell = dict(cell_type=kind, metadata={}, id=f"phase1-{len(cells):02d}",
                source=dedent(source).strip() + "\n")
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def md(source):
    add("markdown", source)


def code(source):
    add("code", source)


md(r'''
# Phase 1 lab — Baseline and catastrophic forgetting

**Research question:** after medical-only fine-tuning, does Qwen2.5-1.5B-Instruct
improve on medical questions while losing general capability?

This notebook implements only Phase 1 of *Project Roadmap (1).pdf*, informed by
*MLTA_2026_Project_Proposal-2 (1).pdf*. It is self-contained: copy this notebook
to another folder or Colab; no project Python files or previous notebook state are needed.
Run cells in order from a fresh kernel. Outputs go under `phase1_outputs/`.

The roadmap places LoRA in Phase 2, while the proposal makes full fine-tuning optional.
For limited compute, the project owner approved **naive LoRA as a labeled Phase 1 substitute**.
Here it means medical-only adaptation of all attention and MLP projection modules,
without replay or selective layers. Full fine-tuning remains an option. These are
different experiments: never report LoRA results as full-fine-tuning results.

There are three modes:

| Mode | Model/data | What it establishes |
|---|---|---|
| `smoke` (default) | Tiny random Qwen-style model, synthetic tasks, CPU | The code executes; **no scientific evidence** |
| `quick` | Qwen2.5-0.5B-Instruct, small real subsets | Downloads and real task integration work; **not the Phase 1 deliverable** |
| `research` | Qwen2.5-1.5B-Instruct, predeclared real subsets | The actual Phase 1 comparison |

**Completion is not guaranteed forgetting.** A result of no clear forgetting is valid.
Do not increase training after inspecting test scores merely to obtain a desired result.
Further experiments need a new configuration and should be labeled exploratory.
''')

md(r'''
## 1. Understand the draft before running the lab

The original `colab.ipynb` launches command-line scripts across all three phases.
`experiments.py` chooses settings, `common.py` prepares prompts/data, `train.py`
optimizes the model, and `evaluate.py` calls EleutherAI's evaluation harness.
`pareto.py` summarizes multiple experiments. Layer analysis, replay, RAG, merging,
and the demo are later-phase work and are excluded here.

We retain the draft's medical prompt, assistant-only training loss, benchmark choices,
and naive LoRA settings (rank 64, learning rate 5e-4, three epochs). We add explicit
data checks, independent setup, configuration-based output isolation, saved individual
predictions, paired comparisons, and a completion audit. We use zero-shot evaluation
explicitly; compare only scores produced by this same protocol, not arbitrary old JSONs.

**Vocabulary.** A tokenizer converts text into integer tokens. A causal language model
predicts the next token. Instruction tuning teaches a chat-formatted model to answer.
LoRA freezes original weights and learns a low-rank update `ΔW = (alpha/r) B A`;
full fine-tuning updates every parameter. An epoch is one pass through the training set.
A learning rate controls update size. Gradient accumulation combines several small
microbatches into one optimizer update. Evaluation measures behavior on held-out tasks.

**Check your understanding:** why can training loss fall while general performance falls too?
Because the optimization objective rewards medical training answers, not retention of
general behavior. Evaluation must therefore remain separate from training.
''')

md(r'''
## 2. Install dependencies and select the kernel

In VS Code, select the project's `.venv/Scripts/python.exe` kernel. In Colab, select
a GPU runtime before a real run. The first rehearsal works on CPU. The code cell below
prints missing packages; set `INSTALL_MISSING = True` only if installation is needed.
Restart the kernel after installation, then run from the top.

PyTorch performs tensor operations; Transformers supplies Qwen and Trainer; PEFT supplies
LoRA; Datasets loads tables; Accelerate supports training; lm-evaluation-harness defines
benchmarks; pandas and matplotlib create tables and plots. There is no RAG, FAISS, Gradio,
or bitsandbytes requirement in this lab. A GPU-enabled PyTorch build must be installed
for your platform if `torch.cuda.is_available()` is false despite having an NVIDIA GPU.
See the [official PyTorch installer](https://pytorch.org/get-started/locally/).

For a real run on Colab, optionally mount Drive **before** choosing `OUTPUT_ROOT` in
the next section and use a persistent directory there. Otherwise download the outputs
before the runtime ends. This notebook does not mount Drive automatically.
''')
code(r'''
import importlib.util
import subprocess
import sys

INSTALL_MISSING = False
packages = {
    "torch": "torch", "transformers": "transformers>=4.45,<6",
    "peft": "peft", "accelerate": "accelerate", "datasets": "datasets",
    "lm_eval": "lm_eval[hf,ifeval]==0.4.13", "pandas": "pandas",
    "matplotlib": "matplotlib", "nbformat": "nbformat", "ipykernel": "ipykernel",
}
missing = [package for module, package in packages.items()
           if importlib.util.find_spec(module) is None]
print("Kernel:", sys.executable)
print("Missing packages:", missing)
if missing and INSTALL_MISSING:
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    raise RuntimeError("Installation finished. Restart the kernel and run from the top.")
if missing:
    raise RuntimeError("Install the listed packages, or set INSTALL_MISSING=True above.")
''')

md(r'''
## 3. Declare the experiment before seeing scores

Start with `MODE = "smoke"`. After it succeeds, restart and use `quick` on a GPU.
Then restart and use `research` for the required 1.5B experiment.
`METHOD = "lora"` is the approved limited-compute substitute. Use `full` only with
sufficient memory. Full training uses float32 weights plus mixed-precision computation;
weights, gradients and Adam states alone are roughly 24 GB for 1.5B parameters, before
activations. A 40 GB-class GPU offers more headroom. Actual usage must be measured.
LoRA saves optimizer memory, but long examples and evaluations can still run out of memory.

Real runs require a CUDA GPU in this lab to avoid accidentally starting a multi-day CPU job.
We fix the seed, sample counts, learning rate, epochs and evaluation protocol in advance.
All real modes use batch size 1 for training plus accumulation; this reduces peak memory.
Changing settings creates a different output directory. `TRIAL` lets you deliberately
start a separate run with otherwise identical settings. Do not reuse a kernel after
changing this configuration: restart and run all cells.
''')
code(r'''
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

MODE = "smoke"                 # smoke | quick | research
METHOD = "lora"                # lora | full
SEED = 0
TRIAL = "01"
OUTPUT_ROOT = Path("phase1_outputs").resolve()
MODEL_REVISION = "main"        # resolved to an immutable commit for a real run
DATA_REVISION = "main"         # set this to the saved training commit for reproduction
assert MODE in {"smoke", "quick", "research"}
assert METHOD in {"lora", "full"}
SMOKE = MODE == "smoke"
CUDA = torch.cuda.is_available()
if not SMOKE and not CUDA:
    raise RuntimeError("Real mode needs a CUDA runtime. Complete smoke locally, then use a GPU.")
torch.set_num_threads(min(4, os.cpu_count() or 1))
set_seed(SEED)
CFG = {
    "protocol": "phase1-v2", "mode": MODE, "method": METHOD, "seed": SEED,
    "trial": TRIAL,
    "model": "local-tiny-random" if SMOKE else (
        "Qwen/Qwen2.5-0.5B-Instruct" if MODE == "quick" else "Qwen/Qwen2.5-1.5B-Instruct"),
    "n_train": 64 if SMOKE else (200 if MODE == "quick" else 10000),
    "epochs": 2 if SMOKE else (1 if MODE == "quick" else (3 if METHOD == "lora" else 2)),
    "lr": 0.01 if SMOKE else (5e-4 if METHOD == "lora" else 2e-5),
    "rank": 4 if SMOKE else 64,
    "max_length": 256 if SMOKE else 384,
    "microbatch": 8 if SMOKE else 1, "grad_accum": 1 if SMOKE else 16,
    "eval_batch": 4 if SMOKE else 1, "num_fewshot": 0,
    "chat_template": True, "bootstrap_replicates": 2000,
    "precision": "float32" if SMOKE else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16"),
}
VERSIONS = {p: metadata.version(p) for p in
            ["torch", "transformers", "peft", "accelerate", "datasets", "lm_eval", "numpy", "pandas"]}
if VERSIONS["lm_eval"] != "0.4.13":
    raise RuntimeError("Use lm_eval[hf,ifeval]==0.4.13 for this notebook's evaluation protocol.")
if not SMOKE:
    from huggingface_hub import HfApi
    api = HfApi()
    CFG["model_revision"] = api.model_info(CFG["model"], revision=MODEL_REVISION).sha
    CFG["training_revision"] = api.dataset_info("openlifescienceai/medmcqa", revision=DATA_REVISION).sha
CFG["versions"] = VERSIONS
CFG["python"] = platform.python_version()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

run_id = fingerprint(CFG)[:12]
ROOT = OUTPUT_ROOT / f"{MODE}_{METHOD}_{run_id}"
ROOT.mkdir(parents=True, exist_ok=True)

def save_json(path, value):
    # Atomic replacement means an interrupted write does not leave half a JSON file.
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)

save_json(ROOT / "configuration.json", CFG)
environment = {"python": sys.version, "platform": platform.platform(), "packages": VERSIONS,
               "cuda": CUDA, "gpu": torch.cuda.get_device_name(0) if CUDA else None}
save_json(ROOT / "environment.json", environment)
(ROOT / "requirements-resolved.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
display(pd.Series(CFG, name="configuration"))
print("Outputs:", ROOT)
print("GPU:", environment["gpu"] or "none (CPU rehearsal)")
''')

md(r'''
## 4. Prepare medical training examples

MedMCQA contains a question, four options, and a zero-based answer index (`cop`).
We use only single-answer examples from its **training** split. The assistant target
is a letter plus option text, such as `B. ...`; this is not a reasoning explanation.
We exclude the dataset's explanation field from the input so it cannot reveal the answer.

The shuffle is seeded and selected row IDs are saved. We remove exact duplicate normalized
questions within the training selection. This is not a complete cross-benchmark contamination
audit: different dataset names do not prove there are no shared questions or pretraining overlap.
Only medical data enters training; no Alpaca/replay examples are used.

Smoke mode uses deliberately artificial organ questions. They are for testing, not medical facts.
Inspect the printed example and confirm the option letter matches the option text.
''')
code(r'''
LETTERS = "ABCD"

def question_prompt(question, options):
    return "Question: " + question + "\n" + "".join(
        f"{letter}. {option}\n" for letter, option in zip(LETTERS, options)) + "Answer:"

def normalize(text):
    return " ".join(text.casefold().split())

def synthetic_rows(n, prefix, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        options = rng.sample(["heart", "lung", "liver", "kidney", "brain", "skin"], 4)
        label = rng.randrange(4)
        rows.append({"id": f"{prefix}-{i}", "question": f"Toy {prefix} case {i}: select {options[label]}.",
                     "options": options, "label": label})
    return rows

if SMOKE:
    rows = synthetic_rows(CFG["n_train"], "training", SEED)
    data_origin = {"kind": "synthetic"}
else:
    from datasets import load_dataset
    raw = load_dataset("openlifescienceai/medmcqa", split="train", revision=CFG["training_revision"])
    data_origin = {"dataset": "openlifescienceai/medmcqa", "split": "train",
                   "revision": CFG["training_revision"], "fingerprint": raw._fingerprint}
    rows, seen = [], set()
    for ex in raw.shuffle(seed=SEED):
        key = normalize(ex["question"])
        if ex.get("choice_type") != "single" or ex["cop"] not in range(4) or key in seen:
            continue
        options = [ex[f"op{x}"] for x in "abcd"]
        if not key or any(not isinstance(x, str) or not x.strip() for x in options):
            continue
        seen.add(key)
        rows.append({"id": str(ex["id"]), "question": ex["question"],
                     "options": options, "label": int(ex["cop"])})
        if len(rows) == CFG["n_train"]:
            break
assert len(rows) == CFG["n_train"], "Not enough valid training examples."

messages = [[{"role": "user", "content": question_prompt(r["question"], r["options"])},
             {"role": "assistant", "content": f"{LETTERS[r['label']]}. {r['options'][r['label']]}"}]
            for r in rows]
print(json.dumps(messages[0], indent=2))
save_json(ROOT / "selected_training_rows.json", rows)
save_json(ROOT / "data_origin.json", data_origin)
''')

md(r'''
## 5. Load the tokenizer and original model checkpoint

Qwen expects a chat template with role and turn markers. Plain text and chat-formatted
text are not interchangeable. We use the model's own tokenizer/template consistently.
Smoke mode constructs a tiny random Qwen2 model locally, including its tokenizer;
it does not download or impersonate the pretrained Qwen model.

The real model revision is a saved commit ID. We will load one model at a time, release
it between evaluation and training, and evaluate base/tuned models at the same precision.
See the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for the
official loading and chat-template example.
''')
code(r'''
def dtype_kwargs(dtype):
    import transformers
    return {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": dtype}

if SMOKE:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config
    BASE = str(ROOT / "tiny_base")
    if not (Path(BASE) / "config.json").exists():
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(
            [m["content"] for pair in messages for m in pair] + ["system user assistant yes no maybe"],
            trainers.BpeTrainer(vocab_size=500,
                                special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>",
                                     pad_token="<|endoftext|>")
        tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
                             "{{ m['content'] }}<|im_end|>\n{% endfor %}"
                             "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        tok.save_pretrained(BASE)
        set_seed(SEED)
        tiny_config = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                 max_position_embeddings=512, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id)
        tiny = AutoModelForCausalLM.from_config(tiny_config)
        tiny.save_pretrained(BASE)
        del tiny
    revision_args = {}
else:
    BASE = CFG["model"]
    revision_args = {"revision": CFG["model_revision"]}
tok = AutoTokenizer.from_pretrained(BASE, **revision_args)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
assert tok.chat_template, "A chat template is required for this protocol."
print(tok.apply_chat_template(messages[0], tokenize=False))
''')

md(r'''
## 6. Tokenize and inspect the training objective

We train only on assistant tokens. Prompt labels and batch-padding labels are `-100`,
which cross-entropy ignores; the attention mask marks padding as invisible. The model
internally shifts labels for next-token prediction. We do not shift them again.

Examples longer than `max_length` are **dropped**, not cut in half. Cutting could remove
the answer. The audit records exactly how many survive and their IDs. This can bias
the dataset toward shorter questions, so report the dropped proportion. Tokenization
at the prompt/answer boundary is checked rather than silently guessed.

Expected output: a nonempty dataset, decoded supervised text containing the answer,
and a length histogram. Answer-only loss is a training objective, not a benchmark score.
''')
code(r'''
def encode_answer_only(pair):
    prefix = tok.apply_chat_template(pair[:-1], tokenize=False, add_generation_prompt=True)
    complete = tok.apply_chat_template(pair, tokenize=False)
    if not complete.startswith(prefix):
        raise ValueError("Template prefix mismatch: inspect the chat template before training.")
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    ids = tok(complete, add_special_tokens=False)["input_ids"]
    if ids[:len(p)] != p:
        raise ValueError("Token boundary mismatch: prompt tokens do not prefix complete chat.")
    if len(ids) > CFG["max_length"]:
        return None
    assert len(ids) > len(p)
    return {"input_ids": ids, "labels": [-100] * len(p) + ids[len(p):]}

encoded = [encode_answer_only(m) for m in messages]
train_data = [e for e in encoded if e is not None]
assert train_data, "All examples were dropped. Inspect lengths and max_length."
audit = {"selected": len(rows), "retained": len(train_data), "dropped": len(rows) - len(train_data),
         "retained_ids": [r["id"] for r, e in zip(rows, encoded) if e is not None],
         "selection_sha256": fingerprint(rows), "encoded_sha256": fingerprint(train_data)}
save_json(ROOT / "training_data_audit.json", audit)
print({k: v for k, v in audit.items() if k != "retained_ids"})
supervised = [x for x in train_data[0]["labels"] if x != -100]
print("Supervised text:", repr(tok.decode(supervised)))

class AnswerCollator:
    def __call__(self, batch):
        width = max(len(e["input_ids"]) for e in batch)
        def pad(key, value):
            return torch.tensor([e[key] + [value] * (width - len(e[key])) for e in batch])
        return {"input_ids": pad("input_ids", tok.pad_token_id), "labels": pad("labels", -100),
                "attention_mask": torch.tensor([[1] * len(e["input_ids"]) +
                                                 [0] * (width - len(e["input_ids"])) for e in batch])}

batch = AnswerCollator()(train_data[:3])
assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
plt.hist([len(e["input_ids"]) for e in train_data], bins=20)
plt.xlabel("Tokens per retained example"); plt.ylabel("Count"); plt.title("Training sequence lengths")
plt.tight_layout(); plt.savefig(ROOT / "training_lengths.png", dpi=150); plt.show()
''')

md(r'''
## 7. Prepare the evaluation protocol

| Benchmark | Capability | Research limit | Metric |
|---|---|---|---|
| MedQA | Medical QA | Full test split | Choice accuracy |
| PubMedQA labeled | QA with an abstract | 500 | yes/no/maybe accuracy |
| MMLU | Broad knowledge (includes medical subjects) | 20 per subject | Size-weighted accuracy |
| HellaSwag | Commonsense completion | 1,000 | Unnormalized choice accuracy |
| IFEval | Following explicit instructions | 150 prompts | Strict instruction-level accuracy |

These limits follow the draft; they are **subsets**, not official full-benchmark results.
MMLU's limit applies per subject, not to the whole group. IFEval can contain several
instructions per prompt. Its primary metric counts instructions, not prompts.
Multiple-choice tasks score candidate likelihoods; IFEval generates text and checks constraints.
IFEval uses its harness generation cap (1,280 new tokens in the installed task).

All runs use zero-shot prompting and the same chat template. We retain the harness's
fixed initial subsets; these are reproducible but not random population samples.
Our bootstrap intervals therefore describe variation within the evaluated sample,
not all possible domain questions or variation between training seeds.

PubMedQA's labeled split is called `train` in the dataset; here it is held out for
evaluation and never used for training. The custom task avoids legacy dataset scripts.
In smoke mode three explicitly named toy tasks replace the benchmark suite, including
a short generation task that checks the generation/evaluation integration.
''')
code(r'''
TASK_DIR = ROOT / "tasks"
TASK_DIR.mkdir(exist_ok=True)
import yaml

if SMOKE:
    TASKS = {"phase1_smoke_medical": ("acc,none", None),
             "phase1_smoke_general": ("acc,none", None),
             "phase1_smoke_generation": ("exact_match,none", None)}
    DOMAIN = ["phase1_smoke_medical"]
    GENERAL = ["phase1_smoke_general", "phase1_smoke_generation"]
    for task, seed in [(DOMAIN[0], 91), (GENERAL[0], 92)]:
        examples = synthetic_rows(12, task, seed)
        if task in GENERAL:
            examples = [{"id": f"general-{i}", "question": f"What is {i} plus one?",
                         "options": [str(i + k) for k in range(4)], "label": 1} for i in range(12)]
        data_path = TASK_DIR / f"{task}.jsonl"
        data_path.write_text("\n".join(json.dumps({"q": question_prompt(r["question"], r["options"]),
                                                "label": r["label"]}) for r in examples), encoding="utf-8")
        task_config = {"task": task, "dataset_path": "json",
                       "dataset_kwargs": {"data_files": {"test": str(data_path)}}, "test_split": "test",
                       "output_type": "multiple_choice", "doc_to_text": "{{q}}",
                       "doc_to_target": "label", "doc_to_choice": list(LETTERS),
                       "num_fewshot": 0,
                       "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]}
        (TASK_DIR / f"{task}.yaml").write_text(yaml.safe_dump(task_config), encoding="utf-8")
    generation_path = TASK_DIR / "generation.jsonl"
    generation_path.write_text("\n".join(json.dumps({"prompt": f"Write the word yes. Example {i}."})
                                         for i in range(3)), encoding="utf-8")
    generation_config = {"task": "phase1_smoke_generation", "dataset_path": "json",
                         "dataset_kwargs": {"data_files": {"test": str(generation_path)}},
                         "test_split": "test", "output_type": "generate_until",
                         "doc_to_text": "prompt", "doc_to_target": "yes", "num_fewshot": 0,
                         "generation_kwargs": {"until": [], "do_sample": False, "max_gen_toks": 8},
                         "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}]}
    (TASK_DIR / "phase1_smoke_generation.yaml").write_text(yaml.safe_dump(generation_config), encoding="utf-8")
else:
    TASKS = {"medqa_4options": ("acc,none", None), "phase1_pubmedqa": ("acc,none", 500),
             "mmlu": ("acc,none", 20), "hellaswag": ("acc,none", 1000),
             "ifeval": ("inst_level_strict_acc,none", 150)}
    if MODE == "quick":
        TASKS = {task: (metric, 2 if task == "mmlu" else 5) for task, (metric, _) in TASKS.items()}
    DOMAIN = ["medqa_4options", "phase1_pubmedqa"]
    GENERAL = ["mmlu", "hellaswag", "ifeval"]
    task_config = {
        "task": "phase1_pubmedqa", "dataset_path": "qiaojin/PubMedQA", "dataset_name": "pqa_labeled",
        "output_type": "multiple_choice", "test_split": "train", "num_fewshot": 0,
        "doc_to_text": "Abstract: {{context.contexts | join('\\n')}}\nQuestion: {{question}}\nAnswer:",
        "doc_to_target": "final_decision", "doc_to_choice": ["yes", "no", "maybe"],
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]}
    (TASK_DIR / "phase1_pubmedqa.yaml").write_text(yaml.safe_dump(task_config), encoding="utf-8")

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
# Index only the required built-in task families; the complete catalog is large.
task_paths = [str(TASK_DIR)]
if not SMOKE:
    builtins = Path(lm_eval.__file__).parent / "tasks"
    task_paths += [str(builtins / p) for p in ["medqa", "mmlu/default", "hellaswag", "ifeval"]]
task_manager = TaskManager(include_path=task_paths, include_defaults=False)
assert all(task in task_manager.all_tasks for task in TASKS), "A required task was not registered."
PROTOCOL = {"tasks": TASKS, "zero_shot": True, "chat_template": tok.chat_template,
            "seed": SEED, "harness": VERSIONS["lm_eval"],
            "model_max_length": 512 if SMOKE else 4096, "selection": "first N per task/subject",
            "custom_tasks": {p.name: p.read_text(encoding="utf-8") for p in TASK_DIR.glob("*.yaml")}}
save_json(ROOT / "evaluation_protocol.json", PROTOCOL)
print("Registered:", list(TASKS))
''')

md(r'''
## 8. Evaluate the original model — before training

We save each completed task immediately, including its individual predictions, document
hashes, resolved task configuration, actual sample counts and metrics. A cache is reused
only when its configuration, data and checkpoint identity match. Changing code that alters
the protocol requires a new protocol version or trial.

The helper below loads the model, calls the harness once per task, and releases memory.
`log_samples=True` is essential: it lets us verify that baseline and tuned scores refer
to the same questions, and later compute paired differences. Never average results from
different models, sample limits or precision settings into one comparison.

Expected output: one score per task and `baseline/<task>.json` files. Smoke accuracy may
be near chance; that is normal for a random model. A real first run downloads model
weights and datasets and may take substantial time, especially IFEval generation.
''')
code(r'''
DEVICE = "cpu" if SMOKE else "cuda"
EVAL_DTYPE = getattr(torch, CFG["precision"])

def release_memory():
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()

def checkpoint_identity(path):
    # Hash actual bytes, not just a folder name, so replaced weights cannot reuse scores.
    digest = hashlib.sha256()
    for file in sorted(Path(path).glob("*.safetensors")):
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()

def load_for_evaluation(tuned=None):
    source = str(tuned) if tuned is not None and METHOD == "full" else BASE
    kwargs = {} if source != BASE else revision_args
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs, **dtype_kwargs(EVAL_DTYPE))
    if tuned is not None and METHOD == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(tuned))
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = True
    return model

def evaluate_checkpoint(label, tuned=None):
    dest = ROOT / label
    dest.mkdir(exist_ok=True)
    model_id = checkpoint_identity(tuned if tuned is not None else BASE) if (tuned is not None or SMOKE) else CFG["model_revision"]
    identity = fingerprint({"cfg": CFG, "protocol": PROTOCOL, "model": model_id,
                            "data": audit["encoded_sha256"], "label": label})
    results, pending = {}, []
    for task in TASKS:
        path = dest / f"{task}.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored["identity"] != identity:
                raise RuntimeError(f"Incompatible cached result: {path}. Use a new TRIAL.")
            results[task] = stored["evaluation"]
        else:
            pending.append(task)
    if pending:
        model = load_for_evaluation(tuned)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=CFG["eval_batch"],
                  max_length=PROTOCOL["model_max_length"])
        try:
            for task in pending:
                started = time.perf_counter()
                result = lm_eval.simple_evaluate(
                    model=lm, tasks=[task], task_manager=task_manager,
                    num_fewshot=0, limit=TASKS[task][1], apply_chat_template=True,
                    log_samples=True, bootstrap_iters=0, random_seed=SEED,
                    numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED)
                if result is None or not result.get("samples"):
                    raise RuntimeError(f"Missing evaluation samples for {task}.")
                # JSON-normalize for identical handling of fresh and resumed results.
                result = json.loads(json.dumps(result, default=str))
                save_json(dest / f"{task}.json", {"identity": identity,
                          "elapsed_seconds": time.perf_counter() - started, "evaluation": result})
                results[task] = result
                table = {**result.get("results", {}), **result.get("groups", {})}
                print(label, task, table[task][TASKS[task][0]])
        finally:
            del lm, model
            release_memory()
    print(label, "complete:", len(results), "tasks")
    return results

baseline_results = evaluate_checkpoint("baseline")
''')

md(r'''
## 9. Train the medical-only model

The naive LoRA setting targets all seven projection types: query, key, value, attention
output, and the three MLP projections. Rank controls the capacity of the added update;
it is not the number of layers. Base weights remain frozen. With `METHOD="full"`, every
parameter is trainable. Neither setting mixes in general data.

The effective batch size on one device is `microbatch × grad_accum`. Mixed precision
reduces computation/memory cost; gradient checkpointing trades extra computation for
lower activation memory. We do not quantize the model in this lab. Reduce the training
microbatch or evaluation batch first if memory is tight; changing scientific settings
requires a new run. Full fine-tuning should not silently switch to LoRA after an error.

Checkpoints are saved each epoch, including optimizer state, to permit interrupted training
to resume. The final completion marker is written only after all model/tokenizer files.
The original draft saved no intermediate checkpoints; a partially completed epoch still
needs to be repeated here. Watch loss and gradient norms for non-finite values.
''')
code(r'''
from transformers.trainer_utils import get_last_checkpoint

TUNED = ROOT / "naive_model"
TRAIN_WORK = ROOT / "training_checkpoints"
train_identity = fingerprint({"cfg": CFG, "data": audit["encoded_sha256"]})
marker = TUNED / "completed.json"
if marker.exists():
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if completion["identity"] != train_identity or completion["weights_sha256"] != checkpoint_identity(TUNED):
        raise RuntimeError("Saved training checkpoint does not match this run. Use a new TRIAL.")
    print("Using completed training:", TUNED)
else:
    set_seed(SEED)
    # Full training uses float32 master weights; LoRA uses the evaluation precision.
    train_dtype = torch.float32 if METHOD == "full" else EVAL_DTYPE
    model = AutoModelForCausalLM.from_pretrained(BASE, **revision_args, **dtype_kwargs(train_dtype))
    if METHOD == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=CFG["rank"], lora_alpha=2 * CFG["rank"], lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
        if not SMOKE:
            model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({100 * trainable / total:.3f}%)")
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(TRAIN_WORK), num_train_epochs=CFG["epochs"],
        per_device_train_batch_size=CFG["microbatch"], gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"], lr_scheduler_type="cosine", warmup_steps=1 if SMOKE else 20,
        optim="adamw_torch", logging_steps=1 if SMOKE else 10,
        save_strategy="epoch", save_total_limit=2, report_to="none",
        seed=SEED, data_seed=SEED, use_cpu=SMOKE,
        bf16=not SMOKE and CFG["precision"] == "bfloat16",
        fp16=not SMOKE and CFG["precision"] == "float16",
        gradient_checkpointing=not SMOKE,
        gradient_checkpointing_kwargs={"use_reentrant": False} if not SMOKE else None,
        dataloader_num_workers=0, dataloader_pin_memory=not SMOKE)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, data_collator=AnswerCollator())
    last = get_last_checkpoint(str(TRAIN_WORK)) if TRAIN_WORK.exists() else None
    started = time.perf_counter()
    if CUDA and not SMOKE:
        torch.cuda.reset_peak_memory_stats()
    outcome = trainer.train(resume_from_checkpoint=last)
    assert np.isfinite(outcome.training_loss), "Non-finite loss; do not evaluate this checkpoint."
    trainer.save_model(str(TUNED))
    tok.save_pretrained(TUNED)
    save_json(ROOT / "training_log.json", trainer.state.log_history)
    save_json(ROOT / "training_summary.json", {
        "metrics": outcome.metrics, "trainable_parameters": trainable, "total_parameters": total,
        "elapsed_seconds_this_session": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if CUDA and not SMOKE else None})
    save_json(marker, {"identity": train_identity, "weights_sha256": checkpoint_identity(TUNED)})
    del trainer, model
    release_memory()
print("Saved", METHOD, "checkpoint:", TUNED)
''')

md(r'''
## 10. Inspect learning, then evaluate the tuned model

The loss plot is a diagnostic. Some fluctuation is normal; a smooth decline is not a
requirement, and a declining loss is not evidence of held-out medical improvement.
The tuned checkpoint now receives exactly the same benchmark protocol as the baseline.
We verify question identities in the following section. If any benchmark fails, fix
that error and rerun the evaluation cell; completed tasks are reused.
''')
code(r'''
history = json.loads((ROOT / "training_log.json").read_text(encoding="utf-8"))
loss_rows = [r for r in history if "loss" in r]
if loss_rows:
    plt.plot([r["step"] for r in loss_rows], [r["loss"] for r in loss_rows], marker=".")
    plt.xlabel("Optimizer update"); plt.ylabel("Assistant-token cross-entropy loss")
    plt.title(f"{MODE}: medical-only {METHOD} training")
    plt.tight_layout(); plt.savefig(ROOT / "training_loss.png", dpi=150); plt.show()
tuned_results = evaluate_checkpoint("tuned", TUNED)
''')

md(r'''
## 11. Compare paired outcomes and uncertainty

For each benchmark, `change_pp = 100 × (tuned − baseline)`. A negative general change
is a forgetting signal. `forgetting_pp = 100 × (baseline − tuned)` uses the opposite sign.
We also report the unweighted mean of the two medical scores and of the three general
scores, following the draft. These composite scores are convenient summaries, not new
validated benchmarks; inspect each task because averaging can conceal opposite trends.

We pair predictions using task, document ID and document hash. We resample the same
question indices for both models. MMLU is resampled within each subject. IFEval resamples
**whole prompts** and counts their correct/total instructions, so instructions in one
prompt are not treated as independent. The 2.5th/97.5th percentiles give an exploratory
paired bootstrap interval. This is more suitable than a universal 'under 3 points is noise'
rule. Intervals reflect sample uncertainty only, not training-seed variation, and are not
adjusted for comparing several tasks. At all-correct/all-wrong boundaries they can be degenerate.

If an interval includes zero, the direction is inconclusive at this precision. An interval
below zero supports a decline on that sample/protocol, but the label **catastrophic** also
requires discussing effect size and practical significance. No single arbitrary cutoff is used.
''')
code(r'''
def sample_records(result, metric):
    name = metric.split(",")[0]
    records = {}
    for task, samples in result["samples"].items():
        for s in samples:
            value = s[name] if name in s else s.get(metric)
            if value is None:
                raise ValueError(f"Missing sample metric {metric}: {task}")
            values = value if isinstance(value, list) else [value]
            assert values and all(isinstance(x, (int, float, bool)) for x in values)
            key = (task, str(s["doc_id"]), s["doc_hash"], s["prompt_hash"], s["target_hash"])
            assert key not in records, "Duplicate sample identity."
            records[key] = (float(sum(values)), len(values))
    return records

def paired_bootstrap(base, tuned, metric, replicates=2000):
    b, t = sample_records(base, metric), sample_records(tuned, metric)
    if set(b) != set(t):
        raise ValueError("Baseline/tuned questions differ. Do not compare these results.")
    keys = sorted(b)
    assert keys, "No paired examples."
    bv, tv = np.array([b[k] for k in keys]), np.array([t[k] for k in keys])
    assert np.array_equal(bv[:, 1], tv[:, 1]), "Metric denominators changed."
    base_score = bv[:, 0].sum() / bv[:, 1].sum()
    tuned_score = tv[:, 0].sum() / tv[:, 1].sum()
    rng = np.random.default_rng(SEED)
    groups = [[i for i, k in enumerate(keys) if k[0] == subject] for subject in sorted({k[0] for k in keys})]
    boot = np.zeros((replicates, 3))
    for indices in groups:
        draw = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        boot[:, 0] += bv[draw, 0].sum(axis=1)
        boot[:, 1] += tv[draw, 0].sum(axis=1)
        boot[:, 2] += bv[draw, 1].sum(axis=1)
    changes = (boot[:, 1] - boot[:, 0]) / boot[:, 2]
    low, high = np.quantile(changes, [0.025, 0.975])
    return base_score, tuned_score, low, high, len(keys), int(bv[:, 1].sum())

comparison = []
for task, (metric, _) in TASKS.items():
    b, t, lo, hi, n, units = paired_bootstrap(baseline_results[task], tuned_results[task], metric,
                                             CFG["bootstrap_replicates"])
    # Check that our sample-level aggregation reproduces the harness's reported score.
    for result, expected in [(baseline_results[task], b), (tuned_results[task], t)]:
        table = {**result.get("results", {}), **result.get("groups", {})}
        assert np.isclose(table[task][metric], expected, atol=1e-8), f"Aggregation mismatch: {task}"
    comparison.append({"task": task, "category": "medical" if task in DOMAIN else "general",
                       "baseline_pct": 100*b, "tuned_pct": 100*t, "change_pp": 100*(t-b),
                       "forgetting_pp": 100*(b-t), "ci_low_pp": 100*lo, "ci_high_pp": 100*hi,
                       "n_prompts": n, "n_scored_units": units})
comparison = pd.DataFrame(comparison).set_index("task")
comparison.to_csv(ROOT / "benchmark_comparison.csv")
display(comparison.round(3))
composites = comparison.groupby("category")[["baseline_pct", "tuned_pct", "change_pp"]].mean()
composites.to_csv(ROOT / "category_summary.csv")
display(composites.round(3))
print("Composite changes are descriptive; no composite confidence interval is claimed.")
''')

md(r'''
## 12. Visualize and write an evidence-based conclusion

The first panel compares absolute scores. The second shows paired changes with bootstrap
intervals. The zero line is no change. General-task intervals wholly below zero indicate
evidence of a decline under this protocol. Medical improvement and general decline are
separate observations: do not assume either happened.

Record a short interpretation: Which medical task improved? Which general task declined?
How large was the effect? Do intervals include zero? Could the answer format itself explain
some IFEval loss? Can one training seed establish that an effect is robust?
''')
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
comparison[["baseline_pct", "tuned_pct"]].plot.barh(ax=axes[0])
axes[0].set_xlabel("Score (%)"); axes[0].set_ylabel("")
axes[0].legend(["Original", f"Naive {METHOD}"])
y = np.arange(len(comparison))
# Draw endpoints directly: percentile intervals need not contain the point estimate.
axes[1].hlines(y, comparison.ci_low_pp, comparison.ci_high_pp, color="steelblue")
axes[1].plot(comparison.change_pp, y, "o", color="navy")
axes[1].set_yticks(y, comparison.index); axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Tuned − original (percentage points)\nPaired 95% bootstrap interval")
fig.suptitle(f"{MODE.upper()} — {METHOD}" + (" — SOFTWARE REHEARSAL ONLY" if SMOKE else ""))
fig.tight_layout(); fig.savefig(ROOT / "phase1_comparison.png", dpi=160, bbox_inches="tight"); plt.show()

general = comparison.loc[GENERAL]
declines = general.index[general.ci_high_pp < 0].tolist()
medical_change = float(composites.loc["medical", "change_pp"])
general_change = float(composites.loc["general", "change_pp"])
if MODE != "research":
    verdict = "Pipeline check only. These results do not establish Phase 1 scientific findings."
elif declines:
    verdict = f"Evidence of general decline on: {', '.join(declines)}. Discuss magnitude before calling it catastrophic."
else:
    verdict = "No clear general decline detected by these paired intervals. This does not prove absence of forgetting."
report = {
    "mode": MODE, "method": METHOD, "model": CFG["model"],
    "medical_mean_change_pp": medical_change, "general_mean_change_pp": general_change,
    "general_tasks_with_intervals_below_zero": declines, "interpretation": verdict,
    "limitations": ["One training seed", "Limited fixed benchmark subsets",
                    "Exploratory intervals, no multiple-comparison correction",
                    "No complete cross-dataset/pretraining contamination audit",
                    "LoRA is a substitute for full fine-tuning" if METHOD == "lora" else "Full fine-tuning",
                    "Answer-format adaptation may contribute to instruction-following changes"],
}
save_json(ROOT / "interpretation.json", report)
print(verdict)
print(f"Medical mean change: {medical_change:+.2f} pp; general mean change: {general_change:+.2f} pp")
''')

md(r'''
## 13. Phase 1 completion audit and handoff

The roadmap requires environment setup, the 1.5B model, MedMCQA preparation, all five
benchmarks, original evaluation, naive training, tuned evaluation, comparison and an
assessment of forgetting. A smoke/quick run intentionally fails the **scientific** completion
gate even when every cell executes. `phase1_requirements_complete` requires research mode.
With LoRA it is explicitly completion under the approved substitute, not strict full fine-tuning.

Keep the configuration, resolved package versions, selected training rows, checkpoint,
per-task predictions, tables and figures together. For a reproduction, reuse the saved
model/training-data revisions and packages; benchmark data can evolve, so the saved
document hashes and task configurations are essential. Re-running the same saved run
reuses completed work. Use a new `TRIAL` for a genuine independent rerun.

These results provide the original/naive reference points for Phase 2. Only afterward
should you compare mitigations and inspect layers. Phase 1 does not establish *where*
forgetting happens or which mitigation is best.
''')
code(r'''
requirements = {
    "environment_recorded": (ROOT / "environment.json").exists(),
    "required_1_5b_model": MODE == "research" and CFG["model"] == "Qwen/Qwen2.5-1.5B-Instruct",
    "real_medmcqa_prepared": not SMOKE and bool(train_data),
    "all_five_real_benchmarks": not SMOKE and len(TASKS) == 5,
    "baseline_complete": set(baseline_results) == set(TASKS),
    "naive_training_complete": marker.exists(),
    "tuned_complete": set(tuned_results) == set(TASKS),
    "paired_comparison_saved": (ROOT / "benchmark_comparison.csv").exists(),
    "forgetting_assessed": (ROOT / "interpretation.json").exists(),
}
completion_audit = {"requirements": requirements, "phase1_requirements_complete": all(requirements.values()),
                    "training_method": METHOD, "approved_lora_substitute": METHOD == "lora",
                    "scientific_mode": MODE == "research"}
save_json(ROOT / "completion_audit.json", completion_audit)
display(pd.Series(requirements, name="satisfied"))
print("Phase 1 requirements complete:", completion_audit["phase1_requirements_complete"])
print("Artifacts:", ROOT)
''')

md(r'''
## Troubleshooting and next lab session

| Symptom | Action and meaning |
|---|---|
| Package missing/wrong kernel | Check `sys.executable`; install into that interpreter and restart. |
| CUDA unavailable | Smoke still works. Use a CUDA runtime/build for quick/research; this is not model failure. |
| Network/download error | Restore access to Hugging Face, then rerun the failed cell. Do not substitute synthetic data and call it research. |
| IFEval NLTK resource error | Read the exception's named resource; download it with `nltk.download(...)`, then retry. Do not omit IFEval. |
| CUDA out of memory | Release other models/restart; reduce batch size. If full training cannot fit, explicitly select the approved LoRA substitute in a new run. |
| All examples dropped | Inspect token lengths, chat template and maximum length. Never train on an empty dataset. |
| Non-finite loss | Stop; inspect precision, gradient norms and learning rate. Record any changed training settings as a new experiment. |
| Dataset/prompt identity mismatch | Do not compare scores. Inspect saved configs and sample hashes; rerun both models under one protocol in a new trial. |
| No forgetting | Report that observation and uncertainty. Do not tune on these test scores to force a decline. |
| Colab disconnect | Restore the persistent output folder. Finished tasks are cached; training resumes from the last epoch checkpoint. |

**Your next steps:** explain the printed loss mask in your own words; complete the CPU
rehearsal; run `quick` on your GPU; then declare and execute the `research` configuration.
Use `benchmark_comparison.csv`, `phase1_comparison.png`, and `interpretation.json` to write
the results paragraph, including negative or inconclusive findings.

References: the two supplied project PDFs; [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct);
[MedMCQA dataset](https://huggingface.co/datasets/openlifescienceai/medmcqa);
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness);
[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer);
[PEFT LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora).
''')

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
target = Path(__file__).resolve().parents[1] / "phase1_lab.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {target.name}: {len(cells)} cells")
