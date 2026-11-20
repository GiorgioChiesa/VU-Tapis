#!/usr/bin/env bash
Q=/home/gchie/workspace/wt/focus8/run_files/gpuq/q
WT=/home/gchie/workspace/wt/focus8-s-long
cd $WT
add() { id=$1; pr=$2; shift 2; $Q add --id $id --priority $pr --group exp24 --cwd $WT --log results/exp24/gpuq_$id.log "$@"; }
mkdir -p results/exp24
for f in 0 1 2 3 4; do add exp24_f$f 190 --cpu --cmd "bash run_files/exp24_hybrid.sh fold $f"; done
add exp24_report 180 --cpu --deps exp24_f0 exp24_f1 exp24_f2 exp24_f3 exp24_f4 --cmd "bash run_files/exp24_hybrid.sh report"
add exp24_viewer 170 --cpu --deps exp24_f0 exp24_f1 exp24_f2 exp24_f3 exp24_f4 --cmd "bash run_files/exp24_hybrid.sh viewer"
