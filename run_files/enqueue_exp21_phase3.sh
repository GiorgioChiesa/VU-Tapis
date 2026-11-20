#!/usr/bin/env bash
# EXP-21 phase 3 jobs, added to the gpuq of the focus8 worktree. Run from anywhere:
#   bash run_files/enqueue_exp21_phase3.sh
# Priorities 190..150 (above EXP-19, which is 1..5). Dependencies use absolute globs on the phase 1-2 outputs.
Q=/home/gchie/workspace/wt/focus8/run_files/gpuq/q
WT=/home/gchie/workspace/wt/focus8-p3
L=/home/gchie/workspace/wt/focus8/results/M/fold0
FEAT="count:50:$L/untrimmed/RARP*.npz"
LONG=(); CAUSAL=()
for s in 0 1; do
  for r in a3 sed; do LONG+=("glob:$L/long/${r}_s$s/.done"); CAUSAL+=("glob:$L/long/${r}_causal_s$s/.done"); done
done
cd $WT
add() { id=$1; pr=$2; shift 2; $Q add --id $id --priority $pr --group exp21p3 --cwd $WT --log results/p3/gpuq_$id.log "$@"; }
# GPU: exploratory short-event MS-TCN, seeds 0 and 1 (about 10-20 min each)
for s in 0 1; do
  add exp21_p3_mstcn_s$s $((196 - s)) --mem 8000 --deps "$FEAT" --cmd "bash run_files/exp21_p3.sh mstcn $s {gpu}"
done
# CPU
add exp21_p3_fuse 190 --cpu --deps "$FEAT" "${LONG[@]}" --cmd "bash run_files/exp21_p3.sh fuse"
add exp21_p3_online 189 --cpu --deps "$FEAT" "${CAUSAL[@]}" --cmd "bash run_files/exp21_p3.sh online"
add exp21_p3_clip 188 --cpu --deps exp21_p3_fuse --cmd "bash run_files/exp21_p3.sh clip"
add exp21_p3_confusion 187 --cpu --deps exp21_p3_fuse exp21_p3_online --cmd "bash run_files/exp21_p3.sh confusion"
add exp21_p3_corr 186 --cpu --deps exp21_p3_fuse --cmd "bash run_files/exp21_p3.sh corr"
# final regeneration of the viewer: after all the others, renders whatever exists
add exp21_p3_viewer 150 --cpu --deps exp21_p3_mstcn_s0 exp21_p3_mstcn_s1 exp21_p3_fuse exp21_p3_online exp21_p3_clip \
  exp21_p3_confusion exp21_p3_corr --cmd "bash run_files/exp21_p3.sh viewer"
$Q status | grep exp21_p3
