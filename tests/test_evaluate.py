"""Run evaluate.py for real through lm-eval, on the tiny model and local fake
data shaped like the real benchmarks (multiple choice, a group of tasks like
MMLU, a generation task like IFEval, and our PubMedQA prompt)."""
import json
import sys
from pathlib import Path

import evaluate

T, R = sys.argv[1], Path(sys.argv[2])
D = R / "eval_tasks"
D.mkdir(parents=True, exist_ok=True)

mc = [{"q": f"question {i}", "choices": ["A", "B", "C", "D"], "label": i % 4} for i in range(12)]
pub = [{"question": f"is {i} ?", "context": {"contexts": ["first part", "second part"]},
        "final_decision": ["yes", "no", "maybe"][i % 3]} for i in range(12)]
gen = [{"prompt": f"Write about topic {i} ."} for i in range(4)]
for name, rows in (("mc", mc), ("pub", pub), ("gen", gen)):
    (D / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

head = lambda f: f"dataset_path: json\ndataset_kwargs:\n  data_files:\n    test: {(D / f).resolve()}\ntest_split: test\n"
mc_body = ("output_type: multiple_choice\ndoc_to_text: \"Question: {{q}}\\nAnswer:\"\ndoc_to_target: label\n"
           "doc_to_choice: choices\nmetric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n")
(D / "t_mc.yaml").write_text("task: t_mc\n" + head("mc.jsonl") + mc_body)
(D / "t_sub1.yaml").write_text("task: t_sub1\n" + head("mc.jsonl") + mc_body)
(D / "t_sub2.yaml").write_text("task: t_sub2\n" + head("mc.jsonl") + mc_body)
(D / "_t_group.yaml").write_text("group: t_group\ntask:\n  - t_sub1\n  - t_sub2\n"
                                 "aggregate_metric_list:\n  - metric: acc\n    weight_by_size: True\n")
# our real PubMedQA task file, with only the dataset swapped for the local copy
real = (Path("tasks") / "pubmedqa_parquet.yaml").read_text()
body = real[real.index("output_type"):].replace("test_split: train\n", "")
(D / "t_pub.yaml").write_text("task: t_pub\n" + head("pub.jsonl") + body)
(D / "t_gen.yaml").write_text(
    "task: t_gen\n" + head("gen.jsonl") + "output_type: generate_until\ndoc_to_text: prompt\ndoc_to_target: 0\n"
    "generation_kwargs:\n  until: []\n  do_sample: false\n  max_gen_toks: 8\n"
    "metric_list:\n  - metric: exact_match\n    aggregation: mean\n    higher_is_better: true\n")

evaluate.TASK_DIRS = [str(D)]
evaluate.TASKS = {"t_mc": ("acc,none", None), "t_group": ("acc,none", 5),
                  "t_pub": ("acc,none", 9), "t_gen": ("exact_match,none", 2)}
for name, run in (("base", None), ("lora", R / "lora"), ("full", R / "full")):
    sys.argv = ["evaluate.py", "--name", name, "--model", T, "--tasks", ",".join(evaluate.TASKS),
                "--out", str(R / "eval_results"), "--batch_size", "4"] + (["--run", str(run)] if run else [])
    evaluate.main()
    res = json.loads((R / "eval_results" / f"{name}.json").read_text())
    assert set(res["scores"]) == set(evaluate.TASKS), res["scores"]
    assert all(0 <= v <= 1 for v in res["scores"].values())
    assert res["stderr"]["t_mc"] is not None
    print(f"ok  evaluate.py via lm-eval ({name}):", res["scores"])
evaluate.main()        # second call: everything is done, nothing is recomputed
