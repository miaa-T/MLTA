"""Model merging: move the fine-tuned weights part of the way back to the base.

    merged = base + alpha * (fine_tuned - base)

alpha = 1 is the fine-tuned model, alpha = 0 is the base model. No training.

Example:
  python merge.py --run runs/naive --alpha 0.5 --out runs/naive_merge50
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import common as C


def merge_adapter(run, alpha, out):
    """For LoRA the change is B @ A, so scaling B by alpha scales the change."""
    w = load_file(str(Path(run) / "adapter_model.safetensors"))
    w = {k: (v * alpha if "lora_B" in k else v) for k, v in w.items()}
    out.mkdir(parents=True, exist_ok=True)
    save_file(w, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    shutil.copy(Path(run) / "adapter_config.json", out / "adapter_config.json")


def merge_full(base, run, alpha, out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(run, **C.dtype_kw(torch.float16))
    ref = AutoModelForCausalLM.from_pretrained(base, **C.dtype_kw(torch.float16)).state_dict()
    sd = model.state_dict()
    for k, v in sd.items():
        if v.is_floating_point():
            b = ref[k].float()
            sd[k] = (b + alpha * (v.float() - b)).to(v.dtype)
    model.load_state_dict(sd)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(run).save_pretrained(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=C.BASE_MODEL)
    args = p.parse_args()
    out = Path(args.out)
    if C.is_adapter(args.run):
        merge_adapter(args.run, args.alpha, out)
    else:
        merge_full(args.model, args.run, args.alpha, out)
    src = Path(args.run) / "run_config.json"
    cfg = json.loads(src.read_text()) if src.exists() else {}
    cfg.update(merged_from=args.run, merge_alpha=args.alpha)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"saved {out} (alpha = {args.alpha})")


if __name__ == "__main__":
    main()
