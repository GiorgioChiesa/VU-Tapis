#!/usr/bin/env bash
# EXP-21: once the M0 eval-set extraction (and its evaluation) on GPU1 is over, use GPU1 as the second EXP-21 GPU
# while GPU3 is still held: arm M extraction in reverse order (if not yet sharded), logged to results/phase2_watch.log.
cd "$(dirname "$0")/.."
LOG=results/phase2_watch.log
say() { echo "[$(date "+%F %T")] $*" >> $LOG; }
say "gpu1-reuse watcher start"
while pgrep -f "exp21_extract.sh M0" >/dev/null || pgrep -f "exp21_eval.sh results/M0" >/dev/null || ! [ -e results/M0/fold0/untrimmed/eval/held_out.json ]; do sleep 60; done
if pgrep -f "exp21_extract.sh M " >/dev/null && ! pgrep -f "exp21_extract.sh M 3" >/dev/null \
   && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 500 ]; then
  tmux new-session -d -s exp21_extM_rev1 "bash run_files/exp21_extract.sh M 1 rev"; say "GPU1 free: started M extraction rev on GPU1"
else say "gpu1-reuse: nothing to do"; fi
