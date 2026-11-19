#!/usr/bin/env bash
# EXP-21 evaluation of a prediction dir on inner val and held-out test (CPU; tools/eval_focus8.py).
#   bash run_files/exp21_eval.sh results/M/fold0/untrimmed [splits.json]
set -euo pipefail
cd "$(dirname "$0")/.."
P=${1:?pred dir}; SPL=${2:-results/M/fold0/splits.json}
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
mkdir -p $P/eval
for s in inner_val held_out; do
  $PY -W ignore tools/eval_focus8.py --pred-dir $P --splits $SPL --set $s --out $P/eval/$s.json > $P/eval/$s.log 2>&1
done
