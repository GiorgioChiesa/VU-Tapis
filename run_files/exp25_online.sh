#!/usr/bin/env bash
# EXP-25 steps (run from the gpuq, cwd = the focus8-s-long worktree). Outputs under results/exp25/.
#   bash run_files/exp25_online.sh train <fold> <short_causal|a3_causal> <seed> <gpu> [smoke]
#   bash run_files/exp25_online.sh fold <0-4>     (online evaluation of the fold, CPU)
#   bash run_files/exp25_online.sh report         (tables, Wilcoxon, figures, summary page ~/workspace/viewer/EXP25_online)
#   bash run_files/exp25_online.sh viewer         (timeline viewer with the offline and online hybrids)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
MPY=/home/gchie/workspace/envs/mamba/bin/python
S=/home/gchie/workspace/wt/short-long-context/results/S
OUT=results/exp25
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 PYTHONPATH=.:detectron2:tools
case ${1:?step} in
  train)
    F=${2:?fold}; R=${3:?arm}; SEED=${4:?seed}; GPU=${5:?gpu}
    FEAT=$S/LONG/cv_ft/fold$F/untrimmed/checkpoint_best_steps/all; SPL=results/exp22/fold$F/splits.json
    sm=(); D=$OUT/fold$F/${R}_s$SEED
    [[ ${6:-} == smoke ]] && { sm=(--smoke); D=$OUT/smoke_fold${F}_${R}; rm -rf $D; }
    if [[ $R == a3_causal && $F == 0 ]]; then echo "fold 0 causal L-A3 = EXP-22 run"; exit 0; fi
    mkdir -p $D
    echo "[$(date '+%F %T')] start $D gpu $GPU commit $(git rev-parse --short HEAD)" >> $OUT/runs.txt
    set +e
    if [[ $R == short_causal ]]; then
      CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore tools/train_short_mstcn.py --causal "${sm[@]}" --feat-dir $FEAT --splits $SPL \
        --out $D --seed $SEED >> $D/train.log 2>&1
    else
      CUDA_VISIBLE_DEVICES=$GPU $MPY -W ignore tools/train_long_s.py --causal "${sm[@]}" --feat-dir $FEAT --splits $SPL \
        --out $D --seed $SEED >> $D/train.log 2>&1
    fi
    rc=$?; set -e
    echo "[$(date '+%F %T')] end $D (exit $rc)" >> $OUT/runs.txt
    exit $rc ;;
  fold) CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY -W ignore tools/exp25_online.py --fold ${2:?fold} ;;
  report) CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/exp25_summary.py
    mkdir -p results_summary; cp results/exp25/report.md results_summary/exp25.md ;;
  viewer)
    CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/export_timeline_viewer.py \
      --dets "EXP-24 hybrid (ref short + L-A3 long)=results/exp24/fold{fold}/hybrid_dets_s0.json" \
      --dets "EXP-25 online hybrid=results/exp25/fold{fold}/online_hybrid_dets.json" \
      --out /home/gchie/workspace/viewer/timelines_focus8
    CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/export_exp25_viewer.py --out /home/gchie/workspace/viewer/EXP25_online ;;
  *) echo "unknown step" >&2; exit 2 ;;
esac
