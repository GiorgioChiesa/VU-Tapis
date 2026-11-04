#!/usr/bin/env bash
# Experiment 4: T-DEED-style displacement (offset) head for fine localisation of the events.
# G2 MS-TCN on per-fold fine-tuned features (reference F_base of Experiment 2: outputs/orsi/EXP2/F_base,
# dilate3, not re-run), 5-fold video-level CV x 2 seeds, then eval_untrimmed.py + eval_constraints.py on
# the displaced scores. Resumable: finished jobs leave a .done marker.
#
# Usage (in tmux):
#   bash run_files/exp4_offset.sh "3"             # O1, O2, K5, O3 (ONLY="O K" by default)
#   ONLY=FTO bash run_files/exp4_offset.sh "3"    # the offset arms on top of FT_gstatic (clip + GraSP frame)
# Then: python tools/summarize_exp2.py --exp EXP4
#       python tools/analyze_offsets.py --arms EXP2/F_base EXP4/O1_agnostic3 EXP4/O2_class3 EXP4/K5 EXP4/O3_agnostic5
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
GPUS=(${1:-3})
SLOTS=${SLOTS:-6}   # concurrent jobs per GPU (~1 GB each)
STATIC=outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG_static/all
FRAMES=/home/gchie/workspace/data/orsi_tensors
OUT=outputs/orsi/EXP4
LOGS=outputs/untrimmed_logs/exp4
mkdir -p $OUT $LOGS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=detectron2

TWO="--streams two --static-feat-dir $STATIC"
declare -A ARMS=(
  [O1_agnostic3]="--k 3 --offset-head agnostic"
  [O2_class3]="--k 3 --offset-head class"
  [K5]="--k 5"
  [O3_agnostic5]="--k 5 --offset-head agnostic"
  [FTO1_agnostic3]="$TWO --k 3 --offset-head agnostic"
  [FTO3_agnostic5]="$TWO --k 5 --offset-head agnostic"
)
ONLY=(${ONLY:-O K})   # run only the arms whose name starts with one of these prefixes
jobs=()
for arm in "${!ARMS[@]}"; do
  keep=0; for pre in "${ONLY[@]}"; do [[ $arm == $pre* ]] && keep=1; done
  [ $keep = 1 ] || continue
  for seed in 0 1; do
    for f in 0 1 2 3 4; do jobs+=("$arm/seed$seed/fold$f|${ARMS[$arm]} --seed $seed --cv-fold $f/5|$f"); done
  done
done
IFS=$'\n' jobs=($(printf '%s\n' "${jobs[@]}" | sort)); unset IFS

run_job() {  # run_job <gpu> <name> <args> <fold>
  local gpu=$1 name=$2 extra=$3 fold=$4 dir=$OUT/$2 log=$LOGS/${2//\//_}.log
  local feats=outputs/orsi/LONG/cv_ft/fold$fold/untrimmed/checkpoint_best_steps/all
  [ -f "$dir/.done" ] && return
  $PY -W ignore tools/train_hybrid.py --context G2 --target dilate --feat-dir $feats --frame-dir $FRAMES \
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
