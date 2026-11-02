#!/usr/bin/env bash
# Temporal heads (MS-TCN) on frozen per-second features of full Orsi videos.
# Features: tools/infer_untrimmed.py --features --split all (see CLAUDE.md of the workspace).
# Usage: bash run_files/orsi_temporal.sh [GPUS]   e.g. bash run_files/orsi_temporal.sh "0 1 2 3"
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
GPUS=(${1:-0 1})
FC=outputs/orsi/LONG/control/totale/untrimmed/checkpoint_best_steps/all   # encoder fine-tuned on Orsi train
FG=outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all              # GraSP-trained TAPIS, never saw Orsi
OUT=outputs/orsi/TEMPORAL
LOGS=outputs/untrimmed_logs
mkdir -p $OUT $LOGS

jobs=(
  "control_point|$FC|--target point"
  "control_dilate3|$FC|--target dilate --k 3"
  "control_gauss2|$FC|--target gauss --sigma 2"
  "control_asym3|$FC|--target asym --k 3"
  "control_asym3_s1|$FC|--target asym --k 3 --seed 1"
  "control_asym3_s2|$FC|--target asym --k 3 --seed 2"
  "control_asym3_causal|$FC|--target asym --k 3 --causal"
  "control_gauss2_causal|$FC|--target gauss --sigma 2 --causal"
  "grasp_point|$FG|--target point"
  "grasp_dilate3|$FG|--target dilate --k 3"
  "grasp_gauss2|$FG|--target gauss --sigma 2"
  "grasp_asym3|$FG|--target asym --k 3"
)
# 5-fold video-level cross-validation: only with GraSP features (no Orsi leakage)
for t in "dilate3|--target dilate --k 3" "asym3|--target asym --k 3"; do
  for f in 0 1 2 3 4; do
    jobs+=("cv_grasp_${t%%|*}/fold$f|$FG|${t#*|} --cv-fold $f/5")
  done
done

i=0
for j in "${jobs[@]}"; do
  IFS='|' read -r name feats extra <<< "$j"
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  $PY -W ignore tools/train_temporal.py --feat-dir "$feats" $extra --out "$OUT/$name" --gpu "$gpu" --wandb \
    > "$LOGS/temporal_${name//\//_}.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -L "done:" $LOGS/temporal_*.log || true
