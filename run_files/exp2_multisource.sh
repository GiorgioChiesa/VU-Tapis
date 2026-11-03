#!/usr/bin/env bash
# Experiment 2: do phases and atomic actions (GraSP, SAR-RARP50, ESAD) help Orsi event spotting?
# H* of Experiment 1 (G2 MS-TCN, dilate3), 5-fold video-level CV x 2 seeds, then
# eval_untrimmed.py + eval_constraints.py (G2 + caps + order chain = the recommended pipeline).
# Arms 2.1 (H* itself) and T_base (two streams) are not re-run: they are outputs/orsi/HYBRID/G2_dilate3{,-two}.
# Resumable: finished jobs leave a .done marker.
#
# Usage (in tmux):
#   bash run_files/exp2_multisource.sh aux  "2 3"   # auxiliary predictors: quality on their test sets + Orsi posteriors
#   bash run_files/exp2_multisource.sh grid "2 3"   # arms 2.0, 2.2, 2.3, stack x5, joint x5, fine-tuned x2
#   bash run_files/exp2_multisource.sh static "2 3" # center-frame features of the per-fold fine-tuned encoders (FT_base)
# Then: python tools/summarize_exp2.py ; python tools/analyze_aux_events.py ; python tools/analyze_oracle_phase.py
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/gchie/miniforge3/envs/Vu-Tapis/bin/python}
STAGE=${1:?aux|grid|static}
GPUS=(${2:-2 3})
SLOTS=${SLOTS:-4}   # concurrent jobs per GPU (~1 GB each)
FEATS=outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all
STATIC=outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG_static/all
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

if [ "$STAGE" = static ]; then  # 5 folds x 2 shards, 1 worker per GPU (~11 GB each; 2 do not fit next to the grid)
  sjobs=(); for f in 0 1 2 3 4; do for s in 0 1; do sjobs+=("$f $s"); done; done
  nq=${#GPUS[@]}
  for q in $(seq 0 $((nq - 1))); do
    (
      gpu=${GPUS[$((q % ${#GPUS[@]}))]}
      for i in $(seq $q $nq $(( ${#sjobs[@]} - 1 ))); do
        read -r f s <<< "${sjobs[$i]}"
        run=outputs/orsi/LONG/cv_ft/fold$f
        echo "$(date +%T) gpu$gpu start fold$f shard$s/2"
        $PY -W ignore tools/infer_untrimmed.py --run $run --ckpt checkpoint_best_steps.pyth --split all --features \
          --static --frame-dir $FRAMES --out $run/untrimmed/checkpoint_best_steps_static/all --shard $s/2 --gpu $gpu \
          > $LOGS/ft_static_fold${f}_shard$s.log 2>&1 || echo "FAILED fold$f shard$s"
        echo "$(date +%T) gpu$gpu end   fold$f shard$s/2"
      done
    ) &
  done
  wait
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
  [T_S_all]="--streams two --static-feat-dir $STATIC --aux-mode stack --aux-sources $ALL"  # two streams (best input of Exp. 1) + all
  [FT_gstatic]="--streams two --static-feat-dir $STATIC"   # fine-tuned clip features + GraSP-LONG center frame
  [FT_base]="--streams two --static-feat-dir outputs/orsi/LONG/cv_ft/fold{F}/untrimmed/checkpoint_best_steps_static/all"  # both fine-tuned (needs stage static)
)
ONLY=${ONLY:-}   # run only the arms whose name starts with this prefix (e.g. ONLY=T_ while the grid runs)
jobs=()
for arm in "${!ARMS[@]}"; do
  [[ -n $ONLY && $arm != $ONLY* ]] && continue
  for seed in 0 1; do
    for f in 0 1 2 3 4; do jobs+=("$arm/seed$seed/fold$f|${ARMS[$arm]} --seed $seed --cv-fold $f/5|$f"); done
  done
done
# joint arms first (slowest), then the rest
IFS=$'\n' jobs=($(printf '%s\n' "${jobs[@]}" | sort -t'|' -k1,1 -r)); unset IFS

run_job() {  # run_job <gpu> <name> <args> <fold>
  local gpu=$1 name=$2 extra=$3 fold=$4 dir=$OUT/$2 log=$LOGS/${2//\//_}.log feats=$FEATS
  [ -f "$dir/.done" ] && return
  extra=${extra//\{F\}/$fold}   # per-fold paths
  [[ $name == F* ]] && feats=outputs/orsi/LONG/cv_ft/fold$fold/untrimmed/checkpoint_best_steps/all
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
