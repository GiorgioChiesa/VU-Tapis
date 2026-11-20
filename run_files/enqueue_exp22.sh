#!/usr/bin/env bash
# EXP-22 jobs, added to the gpuq of the focus8 worktree: bash run_files/enqueue_exp22.sh
# Fold 0: L-A3 offline + causal x seeds 0, 1 (priority 199..192), report with the online evaluation, viewer.
# Folds 1-4 (secondary cross-validation check): L-A3 offline x seeds 0, 1, offline report (priority 170..).
Q=/home/gchie/workspace/wt/focus8/run_files/gpuq/q
WT=/home/gchie/workspace/wt/focus8-s-long
cd $WT
add() { id=$1; pr=$2; shift 2; $Q add --id $id --priority $pr --group exp22 --cwd $WT --log results/exp22/gpuq_$id.log "$@"; }
mkdir -p results/exp22
add exp22_in_f0 200 --cpu --cmd "bash run_files/exp22_s_long.sh inputs 0"
pr=199
for r in a3 a3_causal; do
  for s in 0 1; do
    add exp22_f0_${r}_s$s $pr --mem 12000 --deps exp22_in_f0 --cmd "bash run_files/exp22_s_long.sh train 0 $r $s {gpu}"
    pr=$((pr - 1))
  done
done
add exp22_report_f0 190 --cpu --deps exp22_f0_a3_s0 exp22_f0_a3_s1 exp22_f0_a3_causal_s0 exp22_f0_a3_causal_s1 \
  --cmd "bash run_files/exp22_s_long.sh report 0 online"
for f in 1 2 3 4; do
  add exp22_in_f$f 175 --cpu --cmd "bash run_files/exp22_s_long.sh inputs $f"
  for s in 0 1; do
    add exp22_f${f}_a3_s$s $((170 - f)) --mem 12000 --deps exp22_in_f$f --cmd "bash run_files/exp22_s_long.sh train $f a3 $s {gpu}"
  done
  add exp22_report_f$f 160 --cpu --deps exp22_f${f}_a3_s0 exp22_f${f}_a3_s1 --cmd "bash run_files/exp22_s_long.sh report $f"
done
add exp22_viewer 150 --cpu --deps exp22_report_f0 exp22_report_f1 exp22_report_f2 exp22_report_f3 exp22_report_f4 \
  --cmd "bash run_files/exp22_s_long.sh viewer"
$Q status | grep exp22
