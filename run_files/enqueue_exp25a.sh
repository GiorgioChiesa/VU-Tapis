#!/usr/bin/env bash
# EXP-25a jobs: bash run_files/enqueue_exp25a.sh  (10 trainings in two chained lanes, 20 CPU evaluations, report, viewer)
Q=/home/gchie/workspace/wt/focus8/run_files/gpuq/q
WT=/home/gchie/workspace/wt/focus8-s-long
cd $WT
mkdir -p results/exp25a
i=0; prev=("" ""); evals=()
for f in 0 1 2 3 4; do for s in 0 1; do
  id=exp25a_train_f${f}_s${s}; lane=$((i % 2))
  deps=(--deps exp25a_smoke2); [ -n "${prev[$lane]}" ] && deps=(--deps exp25a_smoke2 ${prev[$lane]})
  $Q add --id $id --priority $((199 - i)) --group exp25a --cwd $WT --mem 2500 --share --log results/exp25a/gpuq_$id.log "${deps[@]}" \
    --cmd "bash run_files/exp25a.sh train $f $s {gpu}"
  prev[$lane]=$id; i=$((i + 1))
done; done
for f in 0 1 2 3 4; do for F in 2 4 6 10; do
  id=exp25a_eval_f${f}_F${F}; evals+=($id)
  $Q add --id $id --priority 170 --group exp25a --cwd $WT --cpu --log results/exp25a/gpuq_$id.log \
    --deps exp25a_train_f${f}_s0 exp25a_train_f${f}_s1 --cmd "bash run_files/exp25a.sh eval $f $F"
done; done
$Q add --id exp25a_report --priority 160 --group exp25a --cwd $WT --cpu --log results/exp25a/gpuq_exp25a_report.log --deps "${evals[@]}" --cmd "bash run_files/exp25a.sh report"
$Q add --id exp25a_viewer_tl --priority 140 --group exp25a --cwd $WT --cpu --log results/exp25a/gpuq_exp25a_viewer_tl.log --deps exp25a_report --cmd "bash run_files/exp25a.sh viewer"
$Q add --id exp25a_link --priority 130 --group exp25a --cwd $WT --cpu --log results/exp25a/gpuq_exp25a_link.log --deps exp25a_report --cmd "bash run_files/exp25a.sh link"
$Q status | grep -c exp25a
