#!/usr/bin/env bash
# Driver of EXPERIMENT.md steps 2-4. Two lanes = at most two GPU processes at once.
# Usage: bash results/run_all.sh <lane A|B> <gpu>
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis/detectron2
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
R=/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis/outputs/orsi/LONG/control/totale
ENC=/home/gchie/workspace/VU-Tapis/data/pretrained_models/train/LONG.pyth
FD=/home/gchie/workspace/data/orsi_tensors
FB=$R/untrimmed/grasp_LONG/all
FR=$R/untrimmed/grasp_LONG_rgb/all
OUT=/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis/outputs/orsi/TEMPORAL_bgr_rgb
L=results/logs
lane=$1; gpu=$2
case $lane in
  A) shards="0 2"; folds="0 1 2";;
  B) shards="1 3"; folds="3 4";;
esac
ts() { date "+%F %T"; }
echo "$(ts) lane $lane gpu $gpu start (commit $(git rev-parse --short HEAD), dirty: $(git status --porcelain tools | wc -l))"
for s in $shards; do
  t0=$(date +%s)
  $PY -W ignore tools/infer_untrimmed.py --run $R --ckpt checkpoint_best_steps.pyth --split all --shard $s/4 \
    --frame-dir $FD --encoder-ckpt $ENC --normalize --features --flip-channels --out $FR --gpu $gpu \
    > $L/rgb_shard$s.log 2>&1 || { echo "$(ts) shard $s FAILED"; exit 1; }
  echo "$(ts) shard $s done in $(( $(date +%s) - t0 ))s"
done
touch $L/extract_$lane.done
until [ -e $L/extract_A.done ] && [ -e $L/extract_B.done ]; do sleep 30; done
n=$(ls $FR/*.npz | wc -l); echo "$(ts) rgb features: $n npz"
[ "$n" -eq 50 ] || { echo "expected 50 npz"; exit 1; }
for f in $folds; do
  for c in bgr rgb; do
    feats=$FB; [ $c = rgb ] && feats=$FR
    o=$OUT/cv_grasp_dilate3_$c/fold$f
    t0=$(date +%s)
    $PY -W ignore tools/train_temporal.py --feat-dir $feats --target dilate --k 3 --cv-fold $f/5 --seed 0 \
      --out $o --gpu $gpu > $L/train_${c}_fold$f.log 2>&1 || { echo "$(ts) train $c fold$f FAILED"; exit 1; }
    t1=$(date +%s)
    $PY -W ignore tools/eval_untrimmed.py --pred-dir $o/val --frame-dir $FD > $L/eval_${c}_fold$f.log 2>&1 \
      || { echo "$(ts) eval $c fold$f FAILED"; exit 1; }
    $PY -W ignore tools/eval_constraints.py --pred-dir $o/val --frame-dir $FD --alpha 0.1 --phase-smooth 1 \
      --gate-floor 0.2 > $L/constraints_${c}_fold$f.log 2>&1 || { echo "$(ts) constraints $c fold$f FAILED"; exit 1; }
    echo "$(ts) $c fold$f: train $((t1 - t0))s, eval $(( $(date +%s) - t1 ))s"
  done
done
echo "$(ts) lane $lane DONE"
touch $L/lane_$lane.done
