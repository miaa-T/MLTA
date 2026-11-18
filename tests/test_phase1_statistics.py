"""Check paired statistics independently of model execution or network access."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

nb = json.loads((Path(__file__).resolve().parents[1] / "phase1_lab.ipynb").read_text(encoding="utf-8"))
source = next(c["source"] for c in nb["cells"] if "def sample_records(" in c["source"])
tree = ast.parse(source)
functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
scope = {"np": np, "SEED": 0}
exec(compile(functions, "phase1-statistics", "exec"), scope)
paired = scope["paired_bootstrap"]


def result(values, name="acc", subjects=None):
    samples = {}
    for i, value in enumerate(values):
        task = subjects[i] if subjects else "toy"
        samples.setdefault(task, []).append({"doc_id": i, "doc_hash": f"doc-{i}",
                                             "prompt_hash": f"prompt-{i}", "target_hash": f"target-{i}",
                                             name: value})
    return {"samples": samples}


class PairedStatisticsTests(unittest.TestCase):
    def test_identical_predictions_have_zero_paired_change(self):
        base = result([0, 1, 1, 0])
        b, t, low, high, count, units = paired(base, base, "acc,none", 300)
        self.assertEqual((b, t, low, high, count, units), (0.5, 0.5, 0, 0, 4, 4))

    def test_reordering_does_not_change_pairing(self):
        base = result([0, 1, 1, 0])
        tuned = result([0, 1, 1, 0])
        tuned["samples"]["toy"].reverse()
        self.assertEqual(paired(base, tuned, "acc,none", 300)[2:4], (0, 0))

    def test_changed_prompt_rejected_even_with_same_document(self):
        base, tuned = result([0, 1]), result([0, 1])
        tuned["samples"]["toy"][0]["prompt_hash"] = "changed-prompt"
        with self.assertRaisesRegex(ValueError, "questions differ"):
            paired(base, tuned, "acc,none", 100)

    def test_ifeval_uses_instruction_ratio_not_mean_of_prompt_means(self):
        name = "inst_level_strict_acc"
        base = result([[True], [False, False, False]], name)
        tuned = result([[True], [True, True, False]], name)
        b, t, low, high, count, units = paired(base, tuned, name + ",none", 300)
        self.assertEqual((b, t, count, units), (0.25, 0.75, 2, 4))
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)

    def test_mmlu_subjects_are_size_weighted(self):
        base = result([1, 0, 0], subjects=["a", "b", "b"])
        tuned = result([1, 1, 1], subjects=["a", "b", "b"])
        b, t, low, high, _, _ = paired(base, tuned, "acc,none", 100)
        self.assertAlmostEqual(b, 1 / 3)
        self.assertEqual(t, 1)
        self.assertAlmostEqual(low, 2 / 3)
        self.assertAlmostEqual(high, 2 / 3)


if __name__ == "__main__":
    unittest.main()
