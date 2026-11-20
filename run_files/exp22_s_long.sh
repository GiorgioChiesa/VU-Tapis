#!/usr/bin/env bash
# EXP-22 steps (run from the gpuq, cwd = the focus8-s-long worktree). Everything is written under results/exp22/.
#   bash run_files/exp22_s_long.sh inputs <fold>
#   bash run_files/exp22_s_long.sh train <fold> <a3|a3_causal> <seed> <gpu> [smoke]
#   bash run_files/exp22_s_long.sh report <fold> [online]
#   bash run_files/exp22_s_long.sh viewer
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
MPY=/home/gchie/workspace/envs/mamba/bin/python
S=/home/gchie/workspace/wt/short-long-context/results/S
OUT=results/exp22
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 PYTHONPATH=.:detectron2:tools
case ${1:?step} in
  inputs) CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/s_fold_inputs.py --fold ${2:?fold} --out $OUT ;;
  train)
    F=${2:?fold}; R=${3:?arm}; SEED=${4:?seed}; GPU=${5:?gpu}; D=$OUT/fold$F/${R}_s$SEED
    extra=(); [[ $R == *_causal ]] && extra=(--causal); sm=(); [[ ${6:-} == smoke ]] && { sm=(--smoke); D=$OUT/smoke_fold${F}_${R}; rm -rf $D; }
    mkdir -p $D
    echo "[$(date '+%F %T')] start $D gpu $GPU commit $(git rev-parse --short HEAD)" >> $OUT/runs.txt
    set +e
    CUDA_VISIBLE_DEVICES=$GPU $MPY -W ignore tools/train_long_s.py "${extra[@]}" "${sm[@]}" \
      --feat-dir $S/LONG/cv_ft/fold$F/untrimmed/checkpoint_best_steps/all --splits $OUT/fold$F/splits.json \
      --out $D --seed $SEED >> $D/train.log 2>&1
    rc=$?; set -e
    echo "[$(date '+%F %T')] end $D (exit $rc)" >> $OUT/runs.txt
    [ $rc -eq 0 ] || exit $rc
    if [ -z "${sm[*]:-}" ]; then
      for st in inner_val held_out; do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY -W ignore tools/eval_focus8.py --pred-dir $D/$st \
          --splits $OUT/fold$F/splits.json --set $st --out $D/eval/$st.json > $D/eval_$st.log 2>&1
      done
    fi ;;
  report) CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/exp22_report.py --fold ${2:?fold} ${3:+--$3} ;;
  viewer) CUDA_VISIBLE_DEVICES="" $PY -W ignore tools/export_exp22_viewer.py --out /home/gchie/workspace/viewer/EXP22_s_long ;;
  *) echo "unknown step" >&2; exit 2 ;;
esac
