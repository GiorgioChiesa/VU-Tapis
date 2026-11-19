#!/usr/bin/env bash
# EXP-21: 4 Hz untrimmed inference of an arm checkpoint (all heads + 1536-d fp16 latents), resumable per video.
#   bash run_files/exp21_extract.sh M 2 [order]     # order: fwd (held out, inner val, train) | rev (train reversed)
#   PATIENT_SET=eval bash run_files/exp21_extract.sh M0 1   # only held out + inner val
# Two processes (fwd on one GPU, rev on another) meet in the middle; finished videos are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
ARM=${1:?arm}; GPU=${2:?gpu}; ORDER=${3:-fwd}
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=$GPU
RUN=results/$ARM/fold0
mapfile -t PAT < <($PY -c "
import json; s=json.load(open(\"$RUN/splits.json\"))
p=s[\"held_out\"]+s[\"inner_val\"]+([] if \"${PATIENT_SET:-all}\"==\"eval\" else s[\"train\"])
print(\"\n\".join(p if \"$ORDER\"==\"fwd\" else p[::-1]))")
echo "[$(date "+%F %T")] start extract $ARM gpu $GPU order $ORDER commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- tools tapis || echo -dirty)" >> $RUN/runs.txt
$PY -W ignore tools/infer_untrimmed.py --run $RUN --ckpt best.pyth --patients "${PAT[@]}" --fps 4 \
  --label-dir /home/gchie/workspace/data/orsi_labels_revised/v2026-11-19_4hz \
  --frame-dir /home/gchie/workspace/data/orsi_tensors_4fps --out $RUN/untrimmed --amp --features \
  --batch-size 32 --gpu 0 2>&1 | grep --line-buffered -v export_vars >> $RUN/untrimmed_$ORDER.log
echo "[$(date "+%F %T")] end extract $ARM $ORDER (exit ${PIPESTATUS[0]})" >> $RUN/runs.txt
