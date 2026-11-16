"""Where does the model change after fine-tuning?

Two measurements per layer:
  1. weight change:  ||W_tuned - W_base|| / ||W_base||  for every attention
     and MLP matrix.
  2. drift:  how far the layer's output (hidden state) moved, measured as
     1 - cosine similarity between base and tuned, on general prompts and
     on medical prompts.

Example:
  python layer_analysis.py --name naive --run runs/naive

Writes results/analysis/<name>_weights.csv, <name>_drift.csv and two figures."""
import argparse
import json
import re
from pathlib import Path

import torch

import common as C

KEY = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+?)\.(?:lora_[AB]\.)?weight$")


def weight_changes(base, run):
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(base, **C.dtype_kw(torch.float16)).state_dict()
    rows = []
    if C.is_adapter(run):
        from safetensors.torch import load_file
        cfg = json.loads((Path(run) / "adapter_config.json").read_text())
        scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
        w = load_file(str(Path(run) / "adapter_model.safetensors"))
        for ka in (k for k in w if "lora_A" in k):
            m = KEY.search(ka)
            if not m:
                continue
            delta = scale * (w[ka.replace("lora_A", "lora_B")].float() @ w[ka].float())
            layer, block, mod = int(m.group(1)), m.group(2), m.group(3)
            b = ref[f"model.layers.{layer}.{block}.{mod}.weight"].float()
            rows.append((layer, block, mod, delta.norm().item(), b.norm().item()))
    else:
        tuned = AutoModelForCausalLM.from_pretrained(run, **C.dtype_kw(torch.float16)).state_dict()
        for k, b in ref.items():
            m = KEY.search(k)
            if m:
                d = tuned[k].float() - b.float()
                rows.append((int(m.group(1)), m.group(2), m.group(3),
                             d.norm().item(), b.float().norm().item()))
    return sorted(rows)


@torch.no_grad()
def hidden_states(model, tok, prompts, max_len=128):
    """Per prompt: tensor [layers + 1, tokens, hidden] on the CPU."""
    out = []
    for msgs in prompts:
        text = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=max_len).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0] for h in hs]).float().cpu())
    return out


def drift(base, run, prompt_sets):
    """Mean (1 - cosine similarity) per layer, base vs tuned, for each prompt set."""
    res = {}
    if C.is_adapter(run):
        model, tok = C.load_model(base, run)
        for name, prompts in prompt_sets.items():
            tuned = hidden_states(model, tok, prompts)
            with model.disable_adapter():
                ref = hidden_states(model, tok, prompts)
            res[name] = _cos_dist(ref, tuned)
    else:
        model, tok = C.load_model(base)
        ref = {n: hidden_states(model, tok, p) for n, p in prompt_sets.items()}
        del model
        model, tok = C.load_model(base, run)
        for name, prompts in prompt_sets.items():
            res[name] = _cos_dist(ref[name], hidden_states(model, tok, prompts))
    return res


def _cos_dist(ref, tuned):
    per_prompt = [1 - torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(dim=1)
                  for a, b in zip(ref, tuned)]
    return torch.stack(per_prompt).mean(dim=0).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--model", default=C.BASE_MODEL)
    p.add_argument("--n_prompts", type=int, default=48)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results/analysis")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import pandas as pd

    # 1. weight change
    rows = weight_changes(args.model, args.run)
    df = pd.DataFrame(rows, columns=["layer", "block", "module", "delta_norm", "base_norm"])
    df["rel_change"] = df.delta_norm / df.base_norm
    df.to_csv(out / f"{args.name}_weights.csv", index=False)
    if len(df):
        grid = df.pivot(index="module", columns="layer", values="rel_change")
        order = [m for m in C.ATTN_MODULES + C.MLP_MODULES if m in grid.index]
        grid = grid.loc[order]
        fig, ax = plt.subplots(figsize=(11, 3.2))
        im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(order)), order)
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_xlabel("layer")
        ax.set_title(f"{args.name}: relative weight change per matrix")
        fig.colorbar(im, ax=ax, label="||dW|| / ||W||")
        fig.tight_layout()
        fig.savefig(out / f"{args.name}_weights.png", dpi=150)
        print("\nMean relative change by block:")
        print(df.groupby("block").rel_change.mean().to_string())
        print("\nLayers that changed most:")
        print(df.groupby("layer").rel_change.mean().sort_values(ascending=False).head(5).to_string())

    # 2. drift of the hidden states
    sets = {
        "general": C.general_examples(args.n_prompts, seed=123, skip=20000, synthetic=args.synthetic),
        "medical": C.medical_examples(args.n_prompts, seed=123, split="validation",
                                      synthetic=args.synthetic),
    }
    d = drift(args.model, args.run, sets)
    dd = pd.DataFrame(d)
    dd.index.name = "layer_output"     # 0 = embeddings, i = output of layer i
    dd.to_csv(out / f"{args.name}_drift.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for col in dd:
        ax.plot(dd.index, dd[col], marker="o", ms=3, label=f"{col} prompts")
    ax.set_xlabel("layer output (0 = embeddings)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("1 - cosine similarity to base")
    ax.set_title(f"{args.name}: how far each layer's output moved")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_drift.png", dpi=150)
    print(f"\nsaved analysis to {out}/")


if __name__ == "__main__":
    main()
