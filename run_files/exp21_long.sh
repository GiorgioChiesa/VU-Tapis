#!/usr/bin/env bash
# EXP-21 long-context models on arm features (resumable; finished runs with .done are skipped).
#   bash run_files/exp21_long.sh M 2            # all 8 runs: {a3, sed} x {offline, causal} x seeds {0, 1}
#   RUNS="a3 sed" SEEDS="0" bash run_files/exp21_long.sh M 2
set -uo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?arm}; GPU=${2:?gpu}
PY=/home/gchie/workspace/envs/mamba/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=$GPU
RUN=results/$ARM/fold0
for r in ${RUNS:-a3 sed a3_causal sed_causal}; do
  for s in ${SEEDS:-0 1}; do
    OUT=$RUN/long/${r}_s$s
    [ -e $OUT/.done ] && continue
    mkdir -p $OUT
    arch=${r%_causal}; extra=(); [[ $r == *_causal ]] && extra=(--causal)
    echo "[$(date "+%F %T")] start $OUT gpu $GPU commit $(git rev-parse --short HEAD)" >> $RUN/long/runs.txt
    $PY -W ignore tools/train_long_mamba.py --arch $arch "${extra[@]}" --feat-dir $RUN/untrimmed \
      --splits $RUN/splits.json --out $OUT --seed $s >> $OUT/train.log 2>&1
    echo "[$(date "+%F %T")] end $OUT (exit $?)" >> $RUN/long/runs.txt
    if [ -e $OUT/.done ]; then  # evaluation (CPU): inner val and held-out predictions of the selected epoch
      for st in inner_val held_out; do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 /home/gchie/miniforge3/envs/Vu-Tapis/bin/python -W ignore tools/eval_focus8.py \
          --pred-dir $OUT/$st --splits $RUN/splits.json --set $st --out $OUT/eval/$st.json > $OUT/eval_$st.log 2>&1
      done
    fi
  done
done
