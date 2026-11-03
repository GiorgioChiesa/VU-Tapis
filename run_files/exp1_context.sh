#!/usr/bin/env bash
# Experiment 1: which global temporal context on top of frozen GraSP-LONG per-second features.
# 5-fold video-level CV on Orsi, tools/train_hybrid.py, then eval_untrimmed.py (and
# eval_constraints.py for G1 = G7 decoding). Resumable: finished jobs leave a .done marker.
#
# Usage (in tmux):
#   bash run_files/exp1_context.sh gate  "2 3"           # G2 dilate3 seed 0: must give 0.458 +/- 0.016
#   bash run_files/exp1_context.sh grid  "2 3"           # offline G0-G5, causal G2-G6, dilate3, seeds 0 1
#   bash run_files/exp1_context.sh asym  "2 3" "G4 G2c"  # asym3 for the given configs (c = causal)
# Then: python tools/summarize_hybrid.py
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
STAGE=${1:?gate|grid|asym}
GPUS=(${2:-2 3})
SLOTS=${SLOTS:-4}   # concurrent jobs per GPU (each needs ~1 GB of GPU memory)
FEATS=outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all
FRAMES=/home/gchie/workspace/data/orsi_tensors
OUT=outputs/orsi/HYBRID
LOGS=outputs/untrimmed_logs/exp1
mkdir -p $OUT $LOGS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

declare -A TARGET=([dilate3]="--target dilate --k 3" [asym3]="--target asym --k 3")

jobs=()
add() {  # add <context> <causal 0|1> <target> <seed>
  local ctx=$1 causal=$2 tgt=$3 seed=$4 c=""
  [ "$causal" = 1 ] && c="c"
  for f in 0 1 2 3 4; do
    jobs+=("${ctx}${c}_${tgt}/seed${seed}/fold${f}|--context $ctx $([ "$causal" = 1 ] && echo --causal) ${TARGET[$tgt]} --seed $seed --cv-fold $f/5")
  done
}
case $STAGE in
  gate) add G2 0 dilate3 0 ;;
  grid)
    for seed in 0 1; do
      for g in G0 G1 G2 G3 G4 G5; do add $g 0 dilate3 $seed; done
      for g in G2 G3 G4 G5 G6; do add $g 1 dilate3 $seed; done
    done ;;
  asym)
    for cfg in ${3:?configs, e.g. "G4 G2c"}; do
      ctx=${cfg%c}; causal=0; [ "$cfg" != "$ctx" ] && causal=1
      for seed in 0 1; do add $ctx $causal asym3 $seed; done
    done ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac

run_job() {  # run_job <gpu> <name> <args>
  local gpu=$1 name=$2 extra=$3 dir=$OUT/$2 log=$LOGS/${2//\//_}.log
  [ -f "$dir/.done" ] && return
  $PY -W ignore tools/train_hybrid.py --feat-dir $FEATS --frame-dir $FRAMES $extra --eval-every 25 \
    --out "$dir" --gpu "$gpu" --wandb > "$log" 2>&1 || { echo "FAILED train $name" >> "$log"; return; }
  $PY -W ignore tools/eval_untrimmed.py --pred-dir "$dir/val" --frame-dir $FRAMES >> "$log" 2>&1 \
    || { echo "FAILED eval $name" >> "$log"; return; }
  if [[ $name == G1_* ]]; then  # G7 = G1 + structured decoding
    $PY -W ignore tools/eval_constraints.py --pred-dir "$dir/val" --frame-dir $FRAMES \
      --alpha 0.1 --phase-smooth 1 --gate-floor 0.2 >> "$log" 2>&1 || { echo "FAILED constraints $name" >> "$log"; return; }
  fi
  touch "$dir/.done"
}

# one sequential queue per (gpu, slot), jobs dealt round-robin
nq=$(( ${#GPUS[@]} * SLOTS ))
for q in $(seq 0 $((nq - 1))); do
  (
    gpu=${GPUS[$((q % ${#GPUS[@]}))]}
    for i in $(seq $q $nq $(( ${#jobs[@]} - 1 ))); do
      IFS='|' read -r name extra <<< "${jobs[$i]}"
      echo "$(date +%T) gpu$gpu start $name"
      run_job "$gpu" "$name" "$extra"
      echo "$(date +%T) gpu$gpu end   $name $([ -f $OUT/$name/.done ] && echo ok || echo FAILED)"
    done
  ) &
done
wait
echo "stage $STAGE finished: $(ls $OUT/*/seed*/fold*/.done 2>/dev/null | wc -l) runs done in total"
