#!/usr/bin/env bash
# EXP-21 fine-tune of one arm on one GPU (resumable: rerun the same command, last.pyth is picked up).
#   bash run_files/exp21_arm.sh M 1     # arm M (teacher heads) on physical GPU 1
#   bash run_files/exp21_arm.sh M0 2    # arm M0 (no teacher heads)
set -euo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?arm M|M0}; GPU=${2:?gpu}
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=$GPU
case $ARM in M) SET=() ;; M0) SET=(--set teacher false) ;; *) echo "unknown arm $ARM" >&2; exit 2 ;; esac
OUT=results/$ARM/fold0
mkdir -p $OUT
echo "[$(date "+%F %T")] start arm $ARM gpu $GPU commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- tools tapis configs || echo -dirty)" >> $OUT/runs.txt
$PY -W ignore tools/train_focus8.py --config configs/Orsi/FOCUS8/exp21_M.yaml --out $OUT "${SET[@]}" 2>&1 \
  | grep --line-buffered -v export_vars >> $OUT/train.log
echo "[$(date "+%F %T")] end arm $ARM (exit ${PIPESTATUS[0]})" >> $OUT/runs.txt
