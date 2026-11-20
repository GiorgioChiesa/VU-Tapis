#!/usr/bin/env bash
# EXP-24 steps (CPU only, run from the gpuq with --cpu, cwd = the focus8-s-long worktree). Outputs under results/exp24/.
#   bash run_files/exp24_hybrid.sh fold <0-4>     (choices on inner val, hybrid, bootstrap; fold 0 also online)
#   bash run_files/exp24_hybrid.sh report         (tables, Wilcoxon, figures, summary page ~/workspace/viewer/EXP24_hybrid)
#   bash run_files/exp24_hybrid.sh events         (per-event strips + mp4 clips, ~/workspace/viewer/EXP24_hybrid/events)
#   bash run_files/exp24_hybrid.sh viewer         (timeline viewer with the hybrid, ~/workspace/viewer/timelines_focus8)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 PYTHONPATH=.:detectron2:tools CUDA_VISIBLE_DEVICES=""
case ${1:?step} in
  fold)
    F=${2:?fold}
    mkdir -p results/exp24
    echo "[$(date '+%F %T')] fold $F commit $(git rev-parse --short HEAD)" >> results/exp24/runs.txt
    extra=()
    [ "$F" = 0 ] && extra=(--online)
    $PY -W ignore tools/exp24_hybrid.py --fold $F "${extra[@]}" ;;
  report)
    $PY -W ignore tools/exp24_summary.py
    mkdir -p results_summary
    cp results/exp24/report.md results_summary/exp24.md ;;
  viewer)
    $PY -W ignore tools/export_timeline_viewer.py --dets "EXP-24 hybrid (ref short + L-A3 long)=results/exp24/fold{fold}/hybrid_dets_s0.json" \
      --out /home/gchie/workspace/viewer/timelines_focus8
    $PY -W ignore tools/export_exp22_viewer.py --out /home/gchie/workspace/viewer/EXP22_s_long ;;
  events)
    $PY -W ignore tools/export_exp24_events.py --out /home/gchie/workspace/viewer/EXP24_hybrid/events ;;
  *) echo "unknown step" >&2; exit 2 ;;
esac
