#!/usr/bin/env bash
# EXP-25a steps (run from the gpuq, cwd = the focus8-s-long worktree). Outputs under results/exp25a/.
#   bash run_files/exp25a.sh train <fold> <seed> <gpu> [smoke]   (short MS-TCN with look-ahead 2,4,6,10 s, features loaded once)
#   bash run_files/exp25a.sh eval <fold> <F>                      (online hybrid evaluation, CPU)
#   bash run_files/exp25a.sh report                               (tables, Wilcoxon, trade-off plot, ~/workspace/viewer/EXP25a_lookahead)
#   bash run_files/exp25a.sh viewer                               (timeline viewer with the best F)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
S=/home/gchie/workspace/wt/short-long-context/results/S
OUT=results/exp25a
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 PYTHONPATH=.:detectron2:tools
case ${1:?step} in
  train)
    F=${2:?fold}; SEED=${3:?seed}; GPU=${4:?gpu}
    FEAT=$S/LONG/cv_ft/fold$F/untrimmed/checkpoint_best_steps/all; SPL=results/exp22/fold$F/splits.json
    sm=(); D=$OUT/fold$F/short_la_s$SEED
    [[ ${5:-} == smoke ]] && { sm=(--smoke); D=$OUT/smoke_fold${F}_s$SEED; rm -rf $D; }
    mkdir -p $D
    echo "[$(date '+%F %T')] start $D gpu $GPU commit $(git rev-parse --short HEAD)" >> $OUT/runs.txt
    set +e
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore tools/train_short_mstcn.py --causal --lookahead 2,4,6,10 "${sm[@]}" --feat-dir $FEAT --splits $SPL \
      --out $D --seed $SEED >> $D/train.log 2>&1
    rc=$?; set -e
    echo "[$(date '+%F %T')] end $D (exit $rc)" >> $OUT/runs.txt
    exit $rc ;;
  eval) CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY -W ignore tools/exp25a_eval.py --fold ${2:?fold} --F ${3:?F} ;;
  report) CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/exp25a_summary.py
    mkdir -p results_summary; cp results/exp25a/report.md results_summary/exp25a.md
    CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/export_exp25a_viewer.py --out /home/gchie/workspace/viewer/EXP25a_lookahead ;;
  viewer)
    BEST=$($PY -c "import json;print(json.load(open('results/exp25a/report.json'))['best_F'])")
    CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/export_timeline_viewer.py \
      --dets "EXP-24 hybrid (ref short + L-A3 long)=results/exp24/fold{fold}/hybrid_dets_s0.json" \
      --dets "EXP-25 online hybrid=results/exp25/fold{fold}/online_hybrid_dets.json" \
      --dets "EXP-25a online hybrid, short look-ahead ${BEST} s=results/exp25a/fold{fold}/eval_F${BEST}/online_hybrid_dets.json" \
      --out /home/gchie/workspace/viewer/timelines_focus8 ;;
  *) echo "unknown step" >&2; exit 2 ;;
esac
