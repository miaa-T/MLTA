"""Fine-tune the model on medical questions (LoRA or full fine-tuning).

Examples:
  python train.py --out runs/lora --rank 16
  python train.py --out runs/replay20 --rank 16 --replay 0.2
  python train.py --out runs/attn_only --target attn
  python train.py --out runs/lower_half --layers lower
  python train.py --out runs/naive --method full --lr 2e-5      (needs a big GPU)
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

import common as C


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.BASE_MODEL)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["lora", "full"], default="lora")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--target", choices=["all", "attn", "mlp"], default="all",
                   help="which blocks get LoRA: attention, MLP or both")
    p.add_argument("--layers", default="all", help="all | lower | upper | e.g. 0-13")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--n_train", type=int, default=10000, help="number of medical examples")
    p.add_argument("--replay", type=float, default=0.0,
                   help="share of general (Alpaca) data in the training mix, e.g. 0.2")
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load_4bit", action="store_true", help="QLoRA, for small GPUs (8 GB)")
    p.add_argument("--grad_ckpt", action="store_true", help="less memory, slower")
    p.add_argument("--synthetic", action="store_true", help="fake data, to test the pipeline")
    p.add_argument("--force", action="store_true", help="retrain even if the run exists")
    return p.parse_args()


def encode(messages, tok, max_len):
    """Token ids for one chat. The loss is computed on the assistant answer only."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    if full.startswith(prompt):
        answer = full[len(prompt):]
    else:
        answer = messages[-1]["content"] + (tok.eos_token or "")
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:
        return None
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def build_data(args, tok):
    med = C.medical_examples(args.n_train, args.seed, synthetic=args.synthetic)
    n_gen = round(len(med) * args.replay / (1 - args.replay)) if args.replay > 0 else 0
    gen = C.general_examples(n_gen, args.seed, synthetic=args.synthetic) if n_gen else []
    data = [e for e in (encode(m, tok, args.max_len) for m in med + gen) if e is not None]
    random.Random(args.seed).shuffle(data)
    print(f"training examples: {len(med)} medical + {len(gen)} general, "
          f"{len(med) + len(gen) - len(data)} dropped as too long")
    return data, len(med), len(gen)


def build_model(args):
    cuda = torch.cuda.is_available()
    kwargs = {"device_map": "auto"} if cuda else {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **C.dtype_kw(torch.float32), **kwargs)
    layers = C.parse_layers(args.layers, args.model)

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.load_4bit:
            model = prepare_model_for_kbit_training(model)
        targets = {"all": C.ATTN_MODULES + C.MLP_MODULES, "attn": C.ATTN_MODULES,
                   "mlp": C.MLP_MODULES}[args.target]
        cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                         target_modules=targets, layers_to_transform=layers,
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    else:
        # full fine-tuning; with --layers only those layers are trained
        if layers is not None:
            for name, prm in model.named_parameters():
                prm.requires_grad = any(f".layers.{i}." in name for i in layers)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"full fine-tuning: {n / 1e6:.0f}M trainable parameters")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def main():
    args = get_args()
    out = Path(args.out)
    if (out / "run_config.json").exists() and not args.force:
        print(f"{out} already trained, skipping (use --force to redo)")
        return
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data, n_med, n_gen = build_data(args, tok)
    model = build_model(args)

    steps = max(1, int(len(data) / (args.batch_size * args.grad_accum) * args.epochs))
    targs = TrainingArguments(
        output_dir=str(out / "tmp"), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * steps)), logging_steps=20, save_strategy="no",
        report_to="none", fp16=torch.cuda.is_available(), seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=data,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    if args.method == "full":
        model = model.to(torch.float16)        # halves the size on disk
    model.save_pretrained(out)
    tok.save_pretrained(out)
    cfg = vars(args) | {"n_medical": n_med, "n_general": n_gen,
                        "final_loss": trainer.state.log_history[-1].get("train_loss")}
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
