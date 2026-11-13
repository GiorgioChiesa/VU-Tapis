#!/usr/bin/env bash
# Experiment 7: one extra worker that joins a running grid of exp7_phase_model.sh (same .done markers and
# atomic mkdir claims in $LOCKS; it never clears $LOCKS). Takes the runs in the given order.
#
# Usage (in tmux, while the grid runs):
#   DEVICE=cuda:0 bash run_files/exp7_extra_worker.sh "A3/0/0 A3/0/1 ... A1/1/4"   # arm/seed/fold
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
JOBS=(${1:?"arm/seed/fold list"})
THREADS=${THREADS:-2}
DEVICE=${DEVICE:-cpu}
OUT=outputs/orsi/EXP7
LOGS=outputs/untrimmed_logs/exp7
LOCKS=$OUT/_locks
mkdir -p $OUT $LOGS $LOCKS
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export PYTHONPATH=${D2:-/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis/detectron2}:$PWD
for j in "${JOBS[@]}"; do
  IFS=/ read -r arm seed f <<< "$j"
  name=$arm/seed$seed/fold$f
  dir=$OUT/$name
  [ -f "$dir/.done" ] && continue
  mkdir "$LOCKS/${name//\//_}" 2>/dev/null || continue
  echo "$(date +%T) x start $name ($DEVICE)"
  $PY -W ignore tools/train_phase.py --arm $arm --seed $seed --cv-fold $f/5 --device $DEVICE --threads $THREADS \
    --feat-dir outputs/orsi/LONG/cv_ft/fold$f/untrimmed/checkpoint_best_steps/all --out $dir \
    > $LOGS/${name//\//_}.log 2>&1 && touch "$dir/.done"
  echo "$(date +%T) x end   $name $([ -f $dir/.done ] && echo ok || echo FAILED)"
done
