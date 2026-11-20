#!/usr/bin/env bash
# EXP-25 training jobs in two chained lanes (at most 2 GPU jobs at a time): bash run_files/enqueue_exp25.sh
Q=/home/gchie/workspace/wt/focus8/run_files/gpuq/q
WT=/home/gchie/workspace/wt/focus8-s-long
cd $WT
mkdir -p results/exp25
jobs=()
for f in 0 1 2 3 4; do for s in 0 1; do jobs+=("short_causal $f $s"); done; done
for f in 1 2 3 4; do for s in 0 1; do jobs+=("a3_causal $f $s"); done; done
i=0; prev=("" "")
for j in "${jobs[@]}"; do
  set -- $j; arm=$1; f=$2; s=$3; id=exp25_${arm}_f${f}_s${s}; lane=$((i % 2))
  deps=(); [ -n "${prev[$lane]}" ] && deps=(--deps ${prev[$lane]})
  $Q add --id $id --priority $((199 - i)) --group exp25 --cwd $WT --mem 12000 --log results/exp25/gpuq_$id.log "${deps[@]}" \
    --cmd "bash run_files/exp25_online.sh train $f $arm $s {gpu}"
  prev[$lane]=$id; i=$((i + 1))
done
$Q status | grep exp25
