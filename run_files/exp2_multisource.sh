#!/usr/bin/env bash
# Experiment 2: do phases and atomic actions (GraSP, SAR-RARP50, ESAD) help Orsi event spotting?
# H* of Experiment 1 (G2 MS-TCN, dilate3), 5-fold video-level CV x 2 seeds, then
# eval_untrimmed.py + eval_constraints.py (G2 + caps + order chain = the recommended pipeline).
# Arm 2.1 (H* itself) is not re-run: it is outputs/orsi/HYBRID/G2_dilate3.
# Resumable: finished jobs leave a .done marker.
#
# Usage (in tmux):
#   bash run_files/exp2_multisource.sh aux  "2 3"   # auxiliary predictors: quality on their test sets + Orsi posteriors
#   bash run_files/exp2_multisource.sh grid "2 3"   # arms 2.0, 2.2, 2.3, stack x5, joint x5, fine-tuned x2
# Then: python tools/summarize_exp2.py ; python tools/analyze_aux_events.py
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
STAGE=${1:?aux|grid}
GPUS=(${2:-2 3})
SLOTS=${SLOTS:-4}   # concurrent jobs per GPU (~1 GB each)
FEATS=outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all
FRAMES=/home/gchie/workspace/data/orsi_tensors
OUT=outputs/orsi/EXP2
LOGS=outputs/untrimmed_logs/exp2
mkdir -p $OUT $LOGS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=detectron2

if [ "$STAGE" = aux ]; then
  for d in grasp sar esad; do
    for mode in eval final; do
      [ -f outputs/orsi/AUX/$mode/$d/aux_report.json ] && continue
      $PY -W ignore tools/train_aux.py --dataset $d --mode $mode --out outputs/orsi/AUX/$mode/$d --gpu ${GPUS[0]} \
        > $LOGS/aux_${mode}_$d.log 2>&1
    done
  done
  exit 0
fi

ALL="grasp_phase grasp_action sar esad"
declare -A ARMS=(
  [2.0_events_only]="--phase-weight 0"
  [2.2_phase_cond]="--phase-cond"
  [2.3_oracle_phase]="--oracle-phase"
  [S_grasp_phase]="--aux-mode stack --aux-sources grasp_phase"
  [S_grasp_action]="--aux-mode stack --aux-sources grasp_action"
  [S_sar]="--aux-mode stack --aux-sources sar"
  [S_esad]="--aux-mode stack --aux-sources esad"
  [S_all]="--aux-mode stack --aux-sources $ALL"
  [J_grasp_phase]="--aux-mode joint --aux-sources grasp_phase"
  [J_grasp_action]="--aux-mode joint --aux-sources grasp_action"
  [J_sar]="--aux-mode joint --aux-sources sar"
  [J_esad]="--aux-mode joint --aux-sources esad"
  [J_all]="--aux-mode joint --aux-sources $ALL"
  [F_base]=""                                            # fine-tuned per-fold features
  [F_S_all]="--aux-mode stack --aux-sources $ALL"        # fine-tuned features + all stacked posteriors
)
jobs=()
for arm in "${!ARMS[@]}"; do
  for seed in 0 1; do
    for f in 0 1 2 3 4; do jobs+=("$arm/seed$seed/fold$f|${ARMS[$arm]} --seed $seed --cv-fold $f/5|$f"); done
  done
done
# joint arms first (slowest), then the rest
IFS=$'\n' jobs=($(printf '%s\n' "${jobs[@]}" | sort -t'|' -k1,1 -r)); unset IFS

run_job() {  # run_job <gpu> <name> <args> <fold>
  local gpu=$1 name=$2 extra=$3 fold=$4 dir=$OUT/$2 log=$LOGS/${2//\//_}.log feats=$FEATS
  [ -f "$dir/.done" ] && return
  [[ $name == F_* ]] && feats=outputs/orsi/LONG/cv_ft/fold$fold/untrimmed/checkpoint_best_steps/all
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
echo "stage $STAGE finished: $(ls $OUT/*/seed*/fold*/.done 2>/dev/null | wc -l) runs done"
