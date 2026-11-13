#!/usr/bin/env bash
# Experiment 7: dedicated phase-only temporal models (A1, A2, A3) on the per-fold fine-tuned features of F_base.
# 5-fold video-level CV x 2 seeds per arm (tools/train_phase.py), then the summary against F_base.
# Resumable: finished runs leave a .done marker; a worker claims a run with an atomic mkdir in $LOCKS.
#
# Usage (in tmux):
#   bash run_files/exp7_phase_model.sh fbase                 # F_base posteriors in the shared format
#   WORKERS=4 THREADS=2 bash run_files/exp7_phase_model.sh grid "A1 A2 A3"   # CPU, 4 x 2 threads
#   DEVICE=cuda:0 WORKERS=4 bash run_files/exp7_phase_model.sh grid "A3"     # one GPU, if approved
#   bash run_files/exp7_phase_model.sh summary
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
STAGE=${1:?fbase|grid|summary}
ARMS=(${2:-A1 A2 A3})
WORKERS=${WORKERS:-4}
THREADS=${THREADS:-2}
DEVICE=${DEVICE:-cpu}
OUT=outputs/orsi/EXP7
LOGS=outputs/untrimmed_logs/exp7
LOCKS=$OUT/_locks
mkdir -p $OUT $LOGS
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export PYTHONPATH=${D2:-/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis/detectron2}:$PWD

if [ "$STAGE" = fbase ]; then
  $PY -W ignore tools/summarize_exp7.py --export-fbase > $LOGS/fbase.log 2>&1 && tail -3 $LOGS/fbase.log
  exit
fi
if [ "$STAGE" = summary ]; then
  $PY -W ignore tools/summarize_exp7.py 2>&1 | tee $LOGS/summary.log
  exit
fi

rm -rf $LOCKS && mkdir -p $LOCKS   # stale claims of an interrupted launch
jobs=()
for arm in "${ARMS[@]}"; do for seed in 0 1; do for f in 0 1 2 3 4; do jobs+=("$arm $seed $f"); done; done; done

worker() {
  for j in "${jobs[@]}"; do
    read -r arm seed f <<< "$j"
    name=$arm/seed$seed/fold$f
    dir=$OUT/$name
    [ -f "$dir/.done" ] && continue
    mkdir "$LOCKS/${name//\//_}" 2>/dev/null || continue
    echo "$(date +%T) w$1 start $name"
    $PY -W ignore tools/train_phase.py --arm $arm --seed $seed --cv-fold $f/5 --device $DEVICE --threads $THREADS \
      --feat-dir outputs/orsi/LONG/cv_ft/fold$f/untrimmed/checkpoint_best_steps/all --out $dir \
      > $LOGS/${name//\//_}.log 2>&1 && touch "$dir/.done"
    echo "$(date +%T) w$1 end   $name $([ -f $dir/.done ] && echo ok || echo FAILED)"
  done
}
for w in $(seq 1 $WORKERS); do worker $w & done
wait
echo "grid finished: $(ls $OUT/*/seed*/fold*/.done 2>/dev/null | wc -l) runs done"
