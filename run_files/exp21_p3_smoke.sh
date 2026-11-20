#!/usr/bin/env bash
# EXP-21 phase 3 smoke test: unit tests and every tool on 1-2 videos, outputs under results/p3_smoke/.
#   bash run_files/exp21_p3_smoke.sh [gpu for the MS-TCN smoke, default 1]
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 PYTHONPATH=.:detectron2:tools
S=results/p3_smoke
mkdir -p $S
CUDA_VISIBLE_DEVICES="" $PY -W ignore tests/test_p3.py && echo TESTS_OK
export CUDA_VISIBLE_DEVICES=""
$PY -W ignore tools/fuse_focus8.py --out $S/fusion --videos 2 --boot 20 && \
$PY -W ignore tools/clip_applier_split.py --dets $S/fusion/final_offline.json --out $S/clip_applier/split.json --videos 2 && \
$PY -W ignore tools/online_eval_focus8.py --out $S/online --videos 2 --boot 20 && \
$PY -W ignore tools/confusion_focus8.py --out $S/confusion --fusion $S/fusion --online $S/online && \
$PY -W ignore tools/corr_focus8.py --out $S/corr --fusion $S/fusion --videos 2 --nshuf 10 && \
CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore tools/train_short_mstcn.py --feat-dir results/M/fold0/untrimmed \
  --splits /home/gchie/workspace/wt/focus8/results/M/fold0/splits.json --out $S/short_mstcn --smoke
$PY -W ignore tools/export_exp21_viewer.py --p3 $S --out $S/viewer && echo VIEWER_OK
echo SMOKE_EXIT $?
