"""The list of experiments, and one command to run them.

  python experiments.py list
  python experiments.py run lora replay20          (train, then evaluate)
  python experiments.py merge naive 0.25 0.5 0.75  (merge, then evaluate)
  python experiments.py run lora --quick --model Qwen/Qwen2.5-0.5B-Instruct
                                                   (fast test of the pipeline)

Every LoRA run changes ONE thing compared with the reference run "lora"."""
import argparse
import subprocess
import sys

import common as C

RUNS = {
    # name:        (extra arguments for train.py, description)
    "naive":      ("--method full --lr 2e-5",       "full fine-tuning, no protection (needs A100)"),
    "naive_lora": ("--rank 64 --lr 5e-4 --epochs 3", "aggressive LoRA: stand-in for 'naive' on a small GPU"),
    "lora":       ("--rank 16",                     "REFERENCE: LoRA rank 16, all blocks, all layers"),
    "lora_r64":   ("--rank 64",                     "higher rank"),
    "lora_lowlr": ("--rank 16 --lr 5e-5",           "lower learning rate"),
    "replay20":   ("--rank 16 --replay 0.2",        "20% general data mixed in"),
    "attn_only":  ("--rank 16 --target attn",       "LoRA on attention only"),
    "mlp_only":   ("--rank 16 --target mlp",        "LoRA on MLP only"),
    "lower_half": ("--rank 16 --layers lower",      "LoRA on the first half of the layers"),
    "upper_half": ("--rank 16 --layers upper",      "LoRA on the second half of the layers"),
    # add your final recipe here once you know what works, for example:
    # "final":    ("--rank 16 --replay 0.2 --target attn", "best combination"),
}


def sh(cmd):
    print("\n>>>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["list", "run", "merge"])
    p.add_argument("names", nargs="*")
    p.add_argument("--model", default=C.BASE_MODEL)
    p.add_argument("--quick", action="store_true", help="tiny run to test the pipeline")
    p.add_argument("--extra", default="", help="more arguments for train.py, e.g. '--load_4bit'")
    args = p.parse_args()
    py = sys.executable
    q = " --quick" if args.quick else ""
    pre = "quick_" if args.quick else ""

    if args.cmd == "list":
        for name, (a, desc) in RUNS.items():
            print(f"{name:11s} {desc:55s} [{a}]")
    elif args.cmd == "run":
        for name in args.names:
            if name == "base":
                sh(f"{py} evaluate.py --name {pre}base --model {args.model}{q}")
                continue
            small = " --n_train 200 --epochs 1" if args.quick else ""
            sh(f"{py} train.py --out runs/{pre}{name} --model {args.model} "
               f"{RUNS[name][0]}{small} {args.extra}")
            sh(f"{py} evaluate.py --name {pre}{name} --run runs/{pre}{name} --model {args.model}{q}")
    elif args.cmd == "merge":
        src, alphas = args.names[0], args.names[1:]
        for a in alphas:
            name = f"{pre}{src}_merge{round(float(a) * 100)}"
            sh(f"{py} merge.py --run runs/{pre}{src} --alpha {a} --out runs/{name} --model {args.model}")
            sh(f"{py} evaluate.py --name {name} --run runs/{name} --model {args.model}{q}")


if __name__ == "__main__":
    main()
