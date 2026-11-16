"""Shared helpers: model loading, prompt formats and datasets."""
import random
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEBUG_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
LETTERS = "ABCD"

# Hugging Face dataset ids
MEDMCQA = "openlifescienceai/medmcqa"          # fine-tuning data
ALPACA = "tatsu-lab/alpaca"                    # general data for replay
MEDQA = "GBaker/MedQA-USMLE-4-options-hf"      # same copy lm-eval uses
TEXTBOOKS = "MedRAG/textbooks"                 # MedQA textbooks, pre-chunked


def mcq_prompt(question, options):
    """Same layout as lm-eval's medqa_4options task, so train and test match."""
    lines = "".join(f"{l}. {o}\n" for l, o in zip(LETTERS, options))
    return f"Question: {question}\n{lines}Answer:"


def chat(user, assistant=None):
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def dtype_kw(dtype):
    """transformers 5 renamed the `torch_dtype` argument to `dtype`."""
    import transformers
    new = int(transformers.__version__.split(".")[0]) >= 5
    return {"dtype" if new else "torch_dtype": dtype}


def is_adapter(run):
    return bool(run) and (Path(run) / "adapter_config.json").exists()


def n_layers(model_name):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_name).num_hidden_layers


def parse_layers(spec, model_name):
    """'all' -> None, 'lower'/'upper' -> half of the layers, '0-5,8' -> list."""
    if spec in (None, "", "all"):
        return None
    if spec in ("lower", "upper"):
        n = n_layers(model_name)
        return list(range(0, n // 2)) if spec == "lower" else list(range(n // 2, n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_model(base, run=None, dtype=None, device_map="auto"):
    """Load base model, optionally with a fine-tuned run on top.

    run can be None (base only), a LoRA adapter folder, or a full model folder.
    Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    full_dir = run if (run and not is_adapter(run)) else base
    tok = AutoTokenizer.from_pretrained(full_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(full_dir, device_map=device_map, **dtype_kw(dtype))
    if is_adapter(run):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, run)
    model.eval()
    return model, tok


# ---------------------------------------------------------------- datasets

def medical_examples(n, seed=0, split="train", synthetic=False):
    """MedMCQA questions as (user, assistant) chat pairs.

    The answer is 'B. <option text>': it starts with the letter, which is
    what the benchmarks score."""
    if synthetic:
        return _synthetic_medical(n, seed)
    from datasets import load_dataset
    ds = load_dataset(MEDMCQA, split=split).shuffle(seed=seed)
    out = []
    for ex in ds:
        if ex.get("choice_type", "single") != "single" or ex["cop"] not in (0, 1, 2, 3):
            continue
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        answer = f"{LETTERS[ex['cop']]}. {opts[ex['cop']]}"
        out.append(chat(mcq_prompt(ex["question"], opts), answer))
        if len(out) >= n:
            break
    return out


def general_examples(n, seed=0, skip=0, synthetic=False):
    """Alpaca instructions as chat pairs. `skip` gives a slice that training never saw."""
    if synthetic:
        return _synthetic_general(n, seed)
    from datasets import load_dataset
    ds = load_dataset(ALPACA, split="train").shuffle(seed=seed)
    out = []
    for i, ex in enumerate(ds):
        if i < skip or not ex["output"].strip():
            continue
        user = ex["instruction"] + (f"\n\n{ex['input']}" if ex["input"].strip() else "")
        out.append(chat(user, ex["output"]))
        if len(out) >= n:
            break
    return out


def _synthetic_medical(n, seed):
    rng = random.Random(seed)
    organs = ["heart", "liver", "kidney", "lung", "brain", "skin"]
    out = []
    for _ in range(n):
        opts = rng.sample(organs, 4)
        k = rng.randrange(4)
        q = f"Which organ is affected in disease {rng.randrange(50)} of the {opts[k]} ?"
        out.append(chat(mcq_prompt(q, opts), f"{LETTERS[k]}. {opts[k]}"))
    return out


def _synthetic_general(n, seed):
    rng = random.Random(seed + 1)
    things = ["a poem", "a list", "a summary", "an email", "a story"]
    return [chat(f"Write {rng.choice(things)} about topic {rng.randrange(50)} .",
                 f"Here is the text about topic {rng.randrange(50)} .") for _ in range(n)]
