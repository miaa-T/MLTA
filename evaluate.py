"""Score one model on the medical and general benchmarks (lm-evaluation-harness).

Examples:
  python evaluate.py --name base
  python evaluate.py --name lora --run runs/lora
  python evaluate.py --name base --quick          (20 questions per task, to test)

Writes results/<name>.json. Finished tasks are saved at once and skipped
on the next call, so a Colab disconnect loses at most one task."""
import argparse
import json
from pathlib import Path

import common as C

# task -> (metric used in the plots, default number of questions; None = all)
TASKS = {
    "medqa_4options":   ("acc,none", None),      # 1,273 questions
    "pubmedqa_parquet": ("acc,none", 500),
    "mmlu":             ("acc,none", 20),        # 20 per subject x 57 subjects
    "hellaswag":        ("acc,none", 1000),
    "ifeval":           ("inst_level_strict_acc,none", 150),
}
DOMAIN = ["medqa_4options", "pubmedqa_parquet"]
GENERAL = ["mmlu", "hellaswag", "ifeval"]
TASK_DIRS = [str(Path(__file__).parent / "tasks")]      # our own lm-eval tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--run", default=None, help="adapter or model folder; empty = base model")
    p.add_argument("--model", default=C.BASE_MODEL)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--batch_size", default="8")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    res.update(name=args.name, run=args.run, model=args.model)
    for key in ("scores", "stderr", "n_limit", "all_metrics"):
        res.setdefault(key, {})
    todo = [t for t in args.tasks.split(",") if t not in res["scores"] or args.quick]
    if not todo:
        print(f"{out} is complete")
        return

    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    full = args.run if (args.run and not C.is_adapter(args.run)) else args.model
    cuda = torch.cuda.is_available()
    lm = HFLM(pretrained=full, peft=args.run if C.is_adapter(args.run) else None,
              dtype="float16" if cuda else "float32", device="cuda" if cuda else "cpu",
              batch_size=args.batch_size)
    tm = TaskManager(include_path=TASK_DIRS)

    for task in todo:
        metric, limit = TASKS[task]
        if args.quick:
            limit = 5 if task == "mmlu" else 20
        r = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=limit, task_manager=tm,
                                    apply_chat_template=True, log_samples=False,
                                    bootstrap_iters=1000)
        table = {**r.get("results", {}), **r.get("groups", {})}
        row = table[task]
        res["scores"][task] = row[metric]
        res["stderr"][task] = row.get(metric.replace(",", "_stderr,"))
        res["n_limit"][task] = limit
        res["all_metrics"][task] = {k: v for k, v in row.items() if isinstance(v, (int, float))}
        print(f"{args.name}  {task}: {row[metric]:.4f}")
        out.write_text(json.dumps(res, indent=1, default=str))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
