#!/usr/bin/env bash
# EXP-21 phase 3 steps (run from the queue, cwd = the focus8-p3 worktree). Reads the phase 1-2 outputs of
# ~/workspace/wt/focus8/results/M/fold0 by absolute path and writes only under results/p3/.
#   bash run_files/exp21_p3.sh fuse | clip | online | confusion | corr | viewer
#   bash run_files/exp21_p3.sh mstcn <seed> <gpu>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 PYTHONPATH=.:detectron2:tools CUDA_VISIBLE_DEVICES=""
M=/home/gchie/workspace/wt/focus8/results/M/fold0
OUT=results/p3
case ${1:?step} in
  fuse)      $PY -W ignore tools/fuse_focus8.py --out $OUT/fusion ;;
  clip)      $PY -W ignore tools/clip_applier_split.py --dets $OUT/fusion/final_offline.json --out $OUT/clip_applier/split.json ;;
  online)    $PY -W ignore tools/online_eval_focus8.py --out $OUT/online ;;
  confusion) $PY -W ignore tools/confusion_focus8.py --out $OUT/confusion --fusion $OUT/fusion --online $OUT/online ;;
  corr)      $PY -W ignore tools/corr_focus8.py --out $OUT/corr --fusion $OUT/fusion ;;
  viewer)    $PY -W ignore tools/export_exp21_viewer.py --out /home/gchie/workspace/viewer/EXP21_focus8 ;;
  mstcn)
    S=${2:?seed}; GPU=${3:?gpu}; D=$OUT/short_mstcn/s$S
    mkdir -p $D
    echo "[$(date '+%F %T')] start $D gpu $GPU commit $(git rev-parse --short HEAD)" >> $OUT/short_mstcn/runs.txt
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore tools/train_short_mstcn.py --feat-dir $M/untrimmed --splits $M/splits.json \
      --out $D --seed $S >> $D/train.log 2>&1
    for st in inner_val held_out; do
      $PY -W ignore tools/eval_focus8.py --pred-dir $D/$st --splits $M/splits.json --set $st --out $D/eval/$st.json > $D/eval_$st.log 2>&1
    done
    echo "[$(date '+%F %T')] end $D" >> $OUT/short_mstcn/runs.txt ;;
  *) echo "unknown step" >&2; exit 2 ;;
esac
