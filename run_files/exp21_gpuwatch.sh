#!/usr/bin/env bash
# EXP-21 GPU watcher: launch arm M0 (tmux exp21_M0) on the first GPU that EXP-19 or EXP-21 M releases.
# GPU 2 / 3 count as released only when their EXP-19 queue lane (run_files/exp19_queue.sh <gpu>) has exited on
# its own (the remaining EXP-19 jobs are held by results/queue/<job>/HOLD_EXP21 claims, so a lane exits after
# its running job) and the GPU has no compute process and < 1500 MiB in use, twice 30 s apart.
# GPU 1 counts once arm M is done. GPU 0 is never used. Nothing of EXP-19 is stopped or touched here.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%F %T")] $*"; }
free() {  # GPU $1 has no compute process and almost no memory in use
  [ -z "$(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader)" ] && \
  [ "$(nvidia-smi --id=$1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]
}
released() {
  case $1 in
    1) [ -e results/M/fold0/.done ] ;;
    2|3) ! pgrep -f "exp19_queue.sh $1" > /dev/null ;;
    *) return 1 ;;
  esac
}
log "watcher start: waiting for GPU 2/3 (EXP-19 lane exit) or GPU 1 (arm M done)"
while true; do
  if [ -e results/M0/fold0/.done ] || tmux has-session -t exp21_M0 2>/dev/null; then log "M0 already running/done"; exit 0; fi
  for g in 2 3 1; do
    if released $g && free $g; then
      sleep 30
      if free $g; then
        log "GPU $g released: launching arm M0 in tmux exp21_M0"
        tmux new-session -d -s exp21_M0 "cd $PWD && bash run_files/exp21_arm.sh M0 $g; exec bash"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
        exit 0
      fi
    fi
  done
  sleep 60
done
