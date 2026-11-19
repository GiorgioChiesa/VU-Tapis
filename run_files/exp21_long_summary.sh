#!/usr/bin/env bash
# EXP-21: markdown summary of the long-context models (inner val and test) with the reference -> results/M/fold0/long/SUMMARY.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
L=results/M/fold0/long; R=results/reference/v2026-11-19; E=results/M/fold0/untrimmed/eval
iv=(); te=()
for r in a3 a3_causal sed sed_causal; do for s in 0 1; do
  [ -e $L/${r}_s$s/eval/held_out.json ] || continue
  iv+=("L-${r} s$s=$L/${r}_s$s/eval/inner_val.json"); te+=("L-${r} s$s=$L/${r}_s$s/eval/held_out.json")
done; done
{ echo "# EXP-21 long-context models on arm M features (fold 0), $(date "+%F %T")"
  echo; echo "## Inner val (5 videos): selection"; $PY tools/summarize_focus8.py "${iv[@]}" "M enc=$E/inner_val.json" --variant nms caps_chain phase_caps_chain
  echo; echo "## Test (10 videos)"; $PY tools/summarize_focus8.py "${te[@]}" "M enc=$E/held_out.json" "S MS-TCN dilate3 s0=$R/dilate3_focus8/eval_test.json" "S MS-TCN dilate3 s1=$R/dilate3_s1_focus8/eval_test.json" --variant nms caps_chain phase_caps_chain --per-class
} > $L/SUMMARY.md
echo "wrote $L/SUMMARY.md"
