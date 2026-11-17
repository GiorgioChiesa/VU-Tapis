#!/usr/bin/env bash
# EXP-19 (short vs long context on revised labels). Usage, from the worktree root, inside tmux:
#   bash run_files/exp19_short_long.sh a0   # step 4: A0 (cv_ft_dilate3) on the revised ground truth, CPU
#   bash run_files/exp19_short_long.sh L    # step 5: arm L, 5 folds, GPUs 2 3
#   bash run_files/exp19_short_long.sh S0   # step 6: arm S pilot, fold 0, GPU 3
# Every step is resumable (.done markers of cv_finetune.py, existing reports are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export PATH=/home/gchie/miniforge3/envs/Vu-Tapis/bin:$PATH
MAIN=/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis
DATA=/home/gchie/workspace/data
LAB1=$DATA/orsi_labels_revised/v2026-11-17
LAB4=$DATA/orsi_labels_revised/v2026-11-17_4hz
FR4=$DATA/orsi_tensors_4fps
SHORT=Grasping_catheter_tip,Clip_or_coagulate_vas_deferens_-_right,Identification_and_clipping_of_SV_arteries_-_right,Metal_clip_on_right_pedicle
LONG=Start_dissection,Incision_of_Denonvilliers_fascia,Transection_of_the_urethra,Stitch_in_bladder
EVAL="--fractional-gt --ignore-dir $LAB1 --classes short:$SHORT long:$LONG"
COMMON=(--ref-run $MAIN/outputs/orsi/LONG/control/totale --init-ckpt $MAIN/data/pretrained_models/train/LONG.pyth
        --set ENDOVIS_DATASET.ADD_IDLE 20 DATA.IDLE_GAP_SEC 30 --targets dilate3 --seeds 0 1)

case "${1:-}" in
  a0)
    for i in 0 1 2 3 4; do
      P=$MAIN/outputs/orsi/TEMPORAL/cv_ft_dilate3/fold$i/val
      R=results/A0/cv_ft_dilate3/fold$i
      mkdir -p $R
      $PY -W ignore tools/eval_untrimmed.py --pred-dir $P --frame-dir $DATA/orsi_tensors --label-dir $LAB1 \
        $EVAL --report-dir $R > $R/eval.log 2>&1
      $PY -W ignore tools/eval_constraints.py --pred-dir $P --frame-dir $DATA/orsi_tensors --label-dir $LAB1 \
        --alpha 0.1 --phase-smooth 1 --gate-floor 0.2 $EVAL --report-dir $R > $R/constraints.log 2>&1
    done
    touch results/A0/.done ;;
  L)
    mkdir -p results/L
    $PY tools/cv_finetune.py --gpus 2 3 --name exp19_L "${COMMON[@]}" --label-dir $LAB1 \
      --out results/L/LONG/cv_ft --temporal-out results/L/TEMPORAL --eval-args "$EVAL" ;;
  S0)
    mkdir -p results/S
    $PY tools/cv_finetune.py --gpus 3 --folds 0 --name exp19_S "${COMMON[@]}" --label-dir $LAB4 \
      --fps 4 --fps-frame-dir $FR4 --temporal-args "--k-sec 3 --crop 16384" \
      --out results/S/LONG/cv_ft --temporal-out results/S/TEMPORAL --eval-args "$EVAL" ;;
  *) echo "usage: $0 a0|L|S0" >&2; exit 2 ;;
esac
