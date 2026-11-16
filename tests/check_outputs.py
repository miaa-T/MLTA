"""Numerical checks on what the smoke test produced."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C
from train import encode

T, R = sys.argv[1], Path(sys.argv[2])
KEYS = lambda sd: [k for k in sd if ".layers." in k and k.endswith("proj.weight")]


def weights(run=None):
    if run is None or C.is_adapter(run):
        m = AutoModelForCausalLM.from_pretrained(T, **C.dtype_kw(torch.float32))
        if run:
            m = PeftModel.from_pretrained(m, run).merge_and_unload()
    else:
        m = AutoModelForCausalLM.from_pretrained(run, **C.dtype_kw(torch.float32))
    return {k: v.clone() for k, v in m.state_dict().items()}


base = weights()

# 1. loss mask: only the answer is trained on
tok = AutoTokenizer.from_pretrained(T)
ex = C.medical_examples(1, synthetic=True)[0]
enc = encode(ex, tok, 256)
kept = tok.decode([t for t in enc["labels"] if t != -100])
assert ex[1]["content"] in kept and "Question" not in kept, kept
assert len(enc["labels"]) == len(enc["input_ids"])
print("ok  loss is computed on the answer only:", repr(kept))

# 2. training lowers the loss
def losses(run):
    return [float(x["loss"]) for x in json.loads((R / run / "train_log.json").read_text()) if "loss" in x]

full, lora = losses("full"), losses("lora")
# the tiny model has a frozen random output layer, so LoRA can only move the loss a little
assert full[-1] < 0.5 * full[0] and lora[-1] < lora[0], (full, lora)
print("ok  training loss goes down: full", [round(x, 2) for x in full], "| lora", [round(x, 2) for x in lora])
cfg = json.loads((R / "lora/run_config.json").read_text())
print("ok  lora run_config:", {k: cfg[k] for k in ("rank", "lr", "n_medical", "n_general")})
assert json.loads((R / "replay/run_config.json").read_text())["n_general"] == 16

# 3. LoRA changes exactly the requested blocks / layers
def changed(run, tol=0.0):
    w = weights(run)
    return {k for k in KEYS(base) if (w[k] - base[k]).abs().max().item() > tol}

assert changed(R / "lora") == set(KEYS(base))
assert all("self_attn" in k for k in changed(R / "attn")) and len(changed(R / "attn")) == 16
assert all(".layers.2." in k or ".layers.3." in k for k in changed(R / "upper")) and changed(R / "upper")
fl = changed(R / "full_lower", tol=1e-3)      # full models are saved in float16
assert fl and all(".layers.0." in k or ".layers.1." in k for k in fl)
print("ok  --target and --layers touch only the requested weights")

# 4. merging: merged = base + 0.5 * (tuned - base)
for kind, tol in (("lora", 1e-6), ("full", 2e-3)):      # full models are saved in float16
    tuned, merged = weights(R / kind), weights(R / f"{kind}_merge50")
    err = max((merged[k] - (base[k] + 0.5 * (tuned[k] - base[k]))).abs().max().item() for k in KEYS(base))
    moved = max((tuned[k] - base[k]).abs().max().item() for k in KEYS(base))
    assert err < tol and moved > 10 * tol, (kind, err, moved)
    print(f"ok  merge {kind}: max error {err:.1e} (weights moved by up to {moved:.1e})")

# 5. layer analysis numbers equal a direct computation
df = pd.read_csv(R / "analysis/lora_weights.csv")
tuned = weights(R / "lora")
for _, r in df.iterrows():
    k = f"model.layers.{r.layer}.{r.block}.{r.module}.weight"
    direct = ((tuned[k] - base[k]).norm() / base[k].norm()).item()
    assert abs(direct - r.rel_change) < 2e-3 * max(direct, 1e-6) + 1e-6, (k, direct, r.rel_change)
assert len(df) == 28
assert set(pd.read_csv(R / "analysis/upper_weights.csv").layer) == {2, 3}
half = pd.read_csv(R / "analysis/lora_merge50_weights.csv")
assert abs((half.rel_change / df.rel_change).mean() - 0.5) < 1e-3
dr = pd.read_csv(R / "analysis/lora_drift.csv")
assert len(dr) == 5 and dr.general.iloc[0] < 1e-6 and dr.general.iloc[-1] > 0
up = pd.read_csv(R / "analysis/upper_drift.csv")
assert up.general.iloc[:3].abs().max() < 1e-6 and up.general.iloc[3:].min() > 0
print("ok  layer analysis matches a direct computation; drift starts where LoRA starts")

# 6. RAG output
g = json.loads((R / "rag/lora.json").read_text())
assert 0 <= g["medqa_rag"] <= 1 and abs(g["rag_gain"] - (g["medqa_rag"] - g["medqa_no_rag"])) < 1e-9
print("ok  rag:", g)

# 7. Pareto table on made-up scores
P = R / "fake_results"
(P / "rag").mkdir(parents=True, exist_ok=True)
fake = {"base": (.40, .60, .55, .50, .45), "naive": (.60, .70, .40, .35, .20),
        "replay20": (.58, .69, .54, .49, .43), "bad": (.39, .55, .30, .30, .20)}
names = ["medqa_4options", "pubmedqa_parquet", "mmlu", "hellaswag", "ifeval"]
for n, s in fake.items():
    (P / f"{n}.json").write_text(json.dumps({"name": n, "scores": dict(zip(names, s))}))
(P / "rag/base.json").write_text(json.dumps({"rag_gain": 0.10}))
subprocess.run([sys.executable, "pareto.py", "--results", str(P)], check=True, capture_output=True)
s = pd.read_csv(P / "summary.csv", index_col=0)
assert dict(s.pareto) == {"naive": True, "replay20": True, "base+RAG": True, "base": False, "bad": False}
assert abs(s.medical["base+RAG"] - 0.55) < 1e-9 and abs(s.general["base+RAG"] - 0.5) < 1e-9
assert (P / "pareto.png").exists()
print("ok  pareto front and +RAG rows")

# 8. lm-eval knows all five tasks, including our own PubMedQA task
from lm_eval.tasks import TaskManager
from evaluate import TASKS
tm = TaskManager(include_path="tasks")
missing = [t for t in TASKS if t not in tm.all_tasks]
assert not missing, missing
print("ok  lm-eval finds:", ", ".join(TASKS))
