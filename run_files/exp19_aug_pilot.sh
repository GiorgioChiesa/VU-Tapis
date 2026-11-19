#!/usr/bin/env bash
# EXP-19 augmentation ablation pilot: arm L (revised labels, 20 Idle per patient, gap 30 s), fold 0 only,
# with ORSI_AUG reduced to subsets. FOLD (default 0) picks the fold, to extend a config to the other folds. Same pipeline and evaluation as arm L of run_files/exp19_short_long.sh;
# the reference (full ORSI_AUG + temporal jitter 2) is results/L fold 0. Usage, from the worktree root, inside tmux:
#   bash run_files/exp19_aug_pilot.sh smoke      # 30-iteration fine-tune of AUG_none, no W&B, prints ORSI_AUG
#   bash run_files/exp19_aug_pilot.sh <config>   # one config: none | geo | geo_photo | geo_surg | geo_jit
#   bash run_files/exp19_aug_pilot.sh chain [deadline]  # the configs in that order, one at a time;
#                                                # no config starts after the deadline (default 2026-11-19 08:00)
# Outputs: results/AUG_<config>/{LONG/cv_ft,TEMPORAL}/... (layout of results/L), results/AUG_<config>/.done
# when fold 0 of the config is finished (.done_fold<k> and run_fold<k>.log for fold k > 0). GPUS (default 2) picks the GPU; cv_finetune.py waits while it is busy and
# its .done markers make every step resumable. EXTRA_SET (default "MAX_SAVE_VIDEO 0") is appended to --set.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
GPUS=${GPUS:-2}
FOLD=${FOLD:-0}
EXTRA_SET=${EXTRA_SET:-MAX_SAVE_VIDEO 0}
export PATH=/home/gchie/miniforge3/envs/Vu-Tapis/bin:$PATH
export NUMPY_MADVISE_HUGEPAGE=0
MAIN=/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis
DATA=/home/gchie/workspace/data
LAB1=$DATA/orsi_labels_revised/v2026-11-17
SHORT=Grasping_catheter_tip,Clip_or_coagulate_vas_deferens_-_right,Identification_and_clipping_of_SV_arteries_-_right,Metal_clip_on_right_pedicle
LONG=Start_dissection,Incision_of_Denonvilliers_fascia,Transection_of_the_urethra,Stitch_in_bladder
EVAL="--fractional-gt --ignore-dir $LAB1 --classes short:$SHORT long:$LONG"
COMMON=(--ref-run $MAIN/outputs/orsi/LONG/control/totale --init-ckpt $MAIN/data/pretrained_models/train/LONG.pyth
        --targets dilate3 --seeds 0 1 --label-dir $LAB1 --folds $FOLD --gpus $GPUS)
# arm L settings (ORSI_AUG.TEMPORAL_JITTER_ROWS is set by each config)
SET_L=(ENDOVIS_DATASET.ADD_IDLE 20 DATA.IDLE_GAP_SEC 30.0 ORSI_AUG.ENABLE True ORSI_AUG.RAND_AUG rand-m5-n2)
# geometric only: flip, crop, rotation at their defaults; every photometric / surgical / erase op off
GEO=(ORSI_AUG.COLOR_P 0.0 "ORSI_AUG.RAND_AUG" "''" ORSI_AUG.BLUR_P 0.0 ORSI_AUG.SMOKE_P 0.0 ORSI_AUG.SPECULAR_P 0.0
     ORSI_AUG.ERASE_P 0.0)
ORDER=(none geo geo_photo geo_surg geo_jit)

aug_set() {  # config -> ORSI_AUG overrides appended after SET_L
  case $1 in
    none)      echo "ORSI_AUG.ENABLE False ORSI_AUG.TEMPORAL_JITTER_ROWS 0" ;;
    geo)       echo "${GEO[*]} ORSI_AUG.TEMPORAL_JITTER_ROWS 0" ;;
    geo_photo) echo "${GEO[*]} ORSI_AUG.COLOR_P 0.8 ORSI_AUG.BLUR_P 0.2 ORSI_AUG.TEMPORAL_JITTER_ROWS 0" ;;
    geo_surg)  echo "${GEO[*]} ORSI_AUG.SMOKE_P 0.2 ORSI_AUG.SPECULAR_P 0.2 ORSI_AUG.TEMPORAL_JITTER_ROWS 0" ;;
    geo_jit)   echo "${GEO[*]} ORSI_AUG.TEMPORAL_JITTER_ROWS 2" ;;
    *) echo "unknown config $1" >&2; return 2 ;;
  esac
}

run_config() {  # $1 = config; writes results/AUG_$1/.done (fold 0) or .done_fold<k> when every step finished
  local c=$1 R=results/AUG_$1 a log mark sfx=""
  [ "$FOLD" != 0 ] && sfx=_fold$FOLD
  mark=$R/.done$sfx
  [ -e $mark ] && { echo "[$(date "+%F %T")] AUG_$c fold $FOLD already done"; return 0; }
  a=$(aug_set "$c")
  mkdir -p $R
  log=$R/run$sfx.log
  echo "[$(date "+%F %T")] start AUG_$c fold $FOLD (gpu $GPUS): $a" | tee -a $log
  # shellcheck disable=SC2086  # the override strings are word lists
  $PY tools/cv_finetune.py --name exp19_AUG_$c "${COMMON[@]}" --set "${SET_L[@]}" $a $EXTRA_SET \
    --out $R/LONG/cv_ft --temporal-out $R/TEMPORAL --eval-args "$EVAL" 2>&1 | tee -a $log
  if grep -q "FAILED" $log || [ ! -e $R/LONG/cv_ft/fold$FOLD/.constraints_dilate3_s1.done ]; then
    echo "[$(date "+%F %T")] AUG_$c FAILED, see $log" | tee -a $log; return 1
  fi
  touch $mark
  echo "[$(date "+%F %T")] end AUG_$c fold $FOLD" | tee -a $log
  WANDB_SILENT=true timeout 600 $PY tools/wandb_runs_tsv.py >/dev/null 2>&1 || echo "wandb_runs.tsv not updated"
}

case "${1:-}" in
  smoke)
    O=results/smoke_aug/none
    rm -rf $O; mkdir -p $O
    # shellcheck disable=SC2046
    setsid $PY tools/cv_finetune.py --name smoke_AUG_none "${COMMON[@]}" --no-wandb --seeds 0 \
      --set "${SET_L[@]}" $(aug_set none) $EXTRA_SET SOLVER.MAX_ITER 30 SOLVER.MAX_EPOCH 1 LOG_PERIOD 10 \
      --out $O/LONG/cv_ft --temporal-out $O/TEMPORAL > $O.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null && [ ! -e $O/LONG/cv_ft/fold$FOLD/.finetune.done ]; do sleep 10; done
    sleep 5; kill -- -$pid 2>/dev/null || true; wait $pid 2>/dev/null || true
    echo "smoke: finetune.done=$([ -e $O/LONG/cv_ft/fold$FOLD/.finetune.done ] && echo yes || echo no)"
    $PY -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print('ORSI_AUG', c['ORSI_AUG']); print('ADD_IDLE', c['ENDOVIS_DATASET']['ADD_IDLE'], 'IDLE_GAP_SEC', c['DATA']['IDLE_GAP_SEC'], 'LABEL_DIR', c['DATA']['LABEL_DIR'])" \
      $O/LONG/cv_ft/fold$FOLD/config.yaml ;;
  chain)
    DEADLINE=${2:-2026-11-19 08:00}
    for c in "${ORDER[@]}"; do
      [ -e results/AUG_$c/.done ] && continue
      if [ "$(date +%s)" -ge "$(date -d "$DEADLINE" +%s)" ]; then
        echo "[$(date "+%F %T")] deadline $DEADLINE reached, AUG_$c not started"; break
      fi
      run_config "$c" || { echo "chain stopped at AUG_$c"; exit 1; }
    done
    echo "[$(date "+%F %T")] chain done" ;;
  none|geo|geo_photo|geo_surg|geo_jit) run_config "$1" ;;
  *) echo "usage: $0 smoke|chain [deadline]|none|geo|geo_photo|geo_surg|geo_jit" >&2; exit 2 ;;
esac
