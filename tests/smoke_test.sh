#!/usr/bin/env bash
# Runs every step on a tiny random model with fake data. No download, no GPU.
# It checks that the code runs, not that the results mean anything.
#   bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
T=tests/tiny; R=tests/out
rm -rf $R $T
python tests/make_tiny_model.py
tr() { python train.py --model $T --synthetic --n_train 64 --epochs 1 --batch_size 8 --grad_accum 1 "$@"; }
tr --out $R/lora --rank 4 --lr 1e-2 --n_train 256 --epochs 3
tr --out $R/replay --rank 4 --replay 0.2
tr --out $R/attn --rank 4 --target attn
tr --out $R/upper --rank 4 --layers upper
tr --out $R/full --method full --lr 3e-3 --n_train 256 --epochs 4
tr --out $R/full_lower --method full --layers lower --lr 1e-2
python merge.py --model $T --run $R/lora --alpha 0.5 --out $R/lora_merge50
python merge.py --model $T --run $R/full --alpha 0.5 --out $R/full_merge50
for n in lora upper full lora_merge50; do
  python layer_analysis.py --model $T --synthetic --n_prompts 4 --name $n --run $R/$n --out $R/analysis
done
python rag.py eval --model $T --synthetic --name base --out $R/rag
python rag.py eval --model $T --synthetic --name lora --run $R/lora --out $R/rag
python tests/check_outputs.py $T $R
python tests/test_evaluate.py $T $R
echo "SMOKE TEST PASSED"
