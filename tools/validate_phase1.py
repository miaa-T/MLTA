"""Execute the standalone smoke lab in an isolated directory with a fresh kernel.

Usage: .venv/Scripts/python.exe tools/validate_phase1.py [--method full]
Only smoke mode is executed; this does not launch a real experiment.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    dest = repo / "tests" / "out_phase1" / args.method
    dest.mkdir(parents=True, exist_ok=True)
    runtime = dest / "jupyter"
    kernel = runtime / "kernels" / "phase1"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Phase 1 validation", "language": "python"}), encoding="utf-8")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    os.environ["HF_HOME"] = str(dest / "hf_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    notebook = nbformat.read(repo / "phase1_lab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if 'MODE = "smoke"' in cell.source:
                cell.source = cell.source.replace('METHOD = "lora"', f'METHOD = "{args.method}"')
            compile(cell.source, "notebook-cell", "exec")
    manager = KernelManager(kernel_name="phase1", kernel_spec_manager=KernelSpecManager(
        kernel_dirs=[str(runtime / "kernels")]))
    client = NotebookClient(notebook, km=manager, timeout=600,
                            resources={"metadata": {"path": str(dest)}},
                            on_cell_start=lambda cell, cell_index: print(
                                f"Cell {cell_index + 1}/{len(notebook.cells)}: {cell.cell_type}", flush=True))
    output = dest / "phase1_lab.executed.ipynb"
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    audits = list((dest / "phase1_outputs").glob("*/completion_audit.json"))
    assert audits
    audit = json.loads(max(audits, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))
    assert not audit["phase1_requirements_complete"]  # Smoke cannot claim scientific completion.
    assert audit["requirements"]["paired_comparison_saved"]
    print("PASS: standalone", args.method, "smoke execution; scientific completion correctly false.")
    print("Executed notebook:", output)


if __name__ == "__main__":
    main()
