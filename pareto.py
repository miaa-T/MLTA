"""Collect all results and draw the trade-off plot.

  python pareto.py

x axis: medical score  = mean of MedQA and PubMedQA
y axis: general score  = mean of MMLU, HellaSwag and IFEval
A model is on the Pareto front if no other model is better on both axes.

"<name>+RAG" rows: MedQA score plus the gain measured by rag.py. The general
score is unchanged, because retrieval is only used for medical questions.

Writes results/summary.csv, results/summary.md and results/pareto.png."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluate import DOMAIN, GENERAL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--baseline", default="base")
    args = p.parse_args()
    root = Path(args.results)

    rows = []
    for f in sorted(root.glob("*.json")):
        r = json.loads(f.read_text())
        if not all(t in r.get("scores", {}) for t in DOMAIN + GENERAL):
            print(f"skipping {f.name}: not all tasks are done")
            continue
        rows.append({"name": r["name"], **r["scores"]})
        rag = root / "rag" / f.name
        if rag.exists():
            g = json.loads(rag.read_text())
            row = {"name": r["name"] + "+RAG", **r["scores"], "rag_gain": g["rag_gain"]}
            row["medqa_4options"] += g["rag_gain"]
            rows.append(row)
    if not rows:
        print("no complete results yet")
        return

    df = pd.DataFrame(rows).set_index("name")
    df["medical"] = df[DOMAIN].mean(axis=1)
    df["general"] = df[GENERAL].mean(axis=1)
    if args.baseline in df.index:
        df["medical_gain"] = df.medical - df.medical[args.baseline]
        df["general_change"] = df.general - df.general[args.baseline]
    df["pareto"] = [not ((df.medical >= a) & (df.general >= b) &
                         ((df.medical > a) | (df.general > b))).any()
                    for a, b in zip(df.medical, df.general)]
    df = df.sort_values("medical", ascending=False)
    df.to_csv(root / "summary.csv")
    (root / "summary.md").write_text((df * 100).round(1).assign(pareto=df.pareto).to_markdown()
                                     if _has_tabulate() else df.to_string())

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    front = df[df.pareto].sort_values("medical")
    ax.plot(front.medical * 100, front.general * 100, "--", color="gray", lw=1,
            label="Pareto front", zorder=1)
    for name, r in df.iterrows():
        color = "black" if name == args.baseline else ("tab:green" if "+RAG" in name else "tab:blue")
        ax.scatter(r.medical * 100, r.general * 100, color=color, s=45, zorder=2)
        ax.annotate(name, (r.medical * 100, r.general * 100), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("medical score (%)  - mean of MedQA, PubMedQA")
    ax.set_ylabel("general score (%)  - mean of MMLU, HellaSwag, IFEval")
    ax.set_title("Medical gain vs. general retention")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "pareto.png", dpi=160)
    print((df * 100).round(1).assign(pareto=df.pareto).to_string())
    print(f"\nsaved {root}/summary.csv, summary.md, pareto.png")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
