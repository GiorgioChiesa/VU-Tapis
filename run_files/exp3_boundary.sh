#!/usr/bin/env bash
# Experiment 3: a phase-change (boundary) head shared by phases and events, for the event classes
# stamped on a phase change. G2 MS-TCN, dilate3, per-fold fine-tuned features (reference F_base of
# Experiment 2: outputs/orsi/EXP2/F_base, not re-run), 5-fold video-level CV x 2 seeds, then
# eval_untrimmed.py + eval_constraints.py. Resumable: finished jobs leave a .done marker.
#
# Usage (in tmux):
#   bash run_files/exp3_boundary.sh "3"            # arms B1..B4 (ONLY=B by default)
#   ONLY=FTB bash run_files/exp3_boundary.sh "3"   # the same on top of FT_gstatic (clip + GraSP frame)
# Then: python tools/summarize_exp2.py --exp EXP3
#       python tools/eval_phase_boundaries.py --arms EXP2/F_base EXP3/B1_head EXP3/B2_add EXP3/B3_transition ...
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
GPUS=(${1:-3})
SLOTS=${SLOTS:-6}   # concurrent jobs per GPU (~1 GB each)
STATIC=outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG_static/all
FRAMES=/home/gchie/workspace/data/orsi_tensors
OUT=outputs/orsi/EXP3
LOGS=outputs/untrimmed_logs/exp3
mkdir -p $OUT $LOGS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=detectron2

TWO="--streams two --static-feat-dir $STATIC"
declare -A ARMS=(
  [B1_head]="--boundary-head"
  [B2_add]="--boundary-head --boundary-fuse add"
  [B3_transition]="--boundary-head --boundary-fuse transition"
  [B4_oracle_boundary]="--oracle-boundary"
  [FTB1_head]="$TWO --boundary-head"
  [FTB2_add]="$TWO --boundary-head --boundary-fuse add"
  [FTB3_transition]="$TWO --boundary-head --boundary-fuse transition"
)
ONLY=${ONLY:-B}   # run only the arms whose name starts with this prefix
jobs=()
for arm in "${!ARMS[@]}"; do
  [[ $arm != $ONLY* ]] && continue
  for seed in 0 1; do
    for f in 0 1 2 3 4; do jobs+=("$arm/seed$seed/fold$f|${ARMS[$arm]} --seed $seed --cv-fold $f/5|$f"); done
  done
done
IFS=$'\n' jobs=($(printf '%s\n' "${jobs[@]}" | sort)); unset IFS

run_job() {  # run_job <gpu> <name> <args> <fold>
  local gpu=$1 name=$2 extra=$3 fold=$4 dir=$OUT/$2 log=$LOGS/${2//\//_}.log
  local feats=outputs/orsi/LONG/cv_ft/fold$fold/untrimmed/checkpoint_best_steps/all
  [ -f "$dir/.done" ] && return
  $PY -W ignore tools/train_hybrid.py --context G2 --target dilate --k 3 --feat-dir $feats --frame-dir $FRAMES \
    $extra --eval-every 25 --out "$dir" --gpu "$gpu" --wandb > "$log" 2>&1 || { echo "FAILED train $name" >> "$log"; return; }
  $PY -W ignore tools/eval_untrimmed.py --pred-dir "$dir/val" --frame-dir $FRAMES >> "$log" 2>&1 \
    || { echo "FAILED eval $name" >> "$log"; return; }
  $PY -W ignore tools/eval_constraints.py --pred-dir "$dir/val" --frame-dir $FRAMES \
    --alpha 0.1 --phase-smooth 1 --gate-floor 0.2 >> "$log" 2>&1 || { echo "FAILED constraints $name" >> "$log"; return; }
  touch "$dir/.done"
}

nq=$(( ${#GPUS[@]} * SLOTS ))
for q in $(seq 0 $((nq - 1))); do
  (
    gpu=${GPUS[$((q % ${#GPUS[@]}))]}
    for i in $(seq $q $nq $(( ${#jobs[@]} - 1 ))); do
      IFS='|' read -r name extra fold <<< "${jobs[$i]}"
      echo "$(date +%T) gpu$gpu start $name"
      run_job "$gpu" "$name" "$extra" "$fold"
      echo "$(date +%T) gpu$gpu end   $name $([ -f $OUT/$name/.done ] && echo ok || echo FAILED)"
    done
  ) &
done
wait
echo "finished: $(ls $OUT/*/seed*/fold*/.done 2>/dev/null | wc -l) runs done"
