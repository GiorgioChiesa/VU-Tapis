#!/usr/bin/env bash
# EXP-21 reference: EXP-19 arm S fold 0 predictions (read-only) re-evaluated on a label version.
# Usage, from the worktree root: bash run_files/exp21_reference.sh v2026-11-19   (CPU only)
# Writes only under results/reference/<version>/. v2026-11-17 is the sanity check (must reproduce EXP-19).
set -euo pipefail
cd "$(dirname "$0")/.."
VER=${1:?label version, e.g. v2026-11-19}
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=""
DATA=/home/gchie/workspace/data
LAB1=$DATA/orsi_labels_revised/$VER
LAB4=$DATA/orsi_labels_revised/${VER}_4hz
S=/home/gchie/workspace/wt/short-long-context/results/S
SHORT=Grasping_catheter_tip,Clip_or_coagulate_vas_deferens_-_right,Identification_and_clipping_of_SV_arteries_-_right,Metal_clip_on_right_pedicle
LONG=Start_dissection,Incision_of_Denonvilliers_fascia,Transection_of_the_urethra,Stitch_in_bladder
EVAL=(--fractional-gt --label-dir $LAB4 --ignore-dir $LAB1 --classes short:$SHORT long:$LONG)
HELD=(RARP35 RARP13 RARP12 RARP40 RARP20 RARP19 RARP49 RARP41 RARP08 RARP07)
OUT=results/reference/$VER
mkdir -p $OUT
R=$OUT/mvit; mkdir -p $R
$PY -W ignore tools/eval_untrimmed.py --pred-dir $S/LONG/cv_ft/fold0/untrimmed/checkpoint_best_steps/all \
  --frame-dir $DATA/orsi_tensors --patients "${HELD[@]}" "${EVAL[@]}" --report-dir $R > $R/eval.log 2>&1
for h in dilate3 dilate3_s1; do
  P=$S/TEMPORAL/cv_ft_$h/fold0/val; R=$OUT/$h; mkdir -p $R
  $PY -W ignore tools/eval_untrimmed.py --pred-dir $P --frame-dir $DATA/orsi_tensors "${EVAL[@]}" --report-dir $R > $R/eval.log 2>&1
  $PY -W ignore tools/eval_constraints.py --pred-dir $P --frame-dir $DATA/orsi_tensors --alpha 0.1 --phase-smooth 1 \
    --gate-floor 0.2 "${EVAL[@]}" --report-dir $R > $R/constraints.log 2>&1
done
touch $OUT/.done
