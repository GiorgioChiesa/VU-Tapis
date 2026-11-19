#!/usr/bin/env bash
# EXP-21 phase 2 watcher (log results/phase2_watch.log). Every 60 s:
#  1. GPU3 free (EXP-19 lane 3 process gone, < 6000 MiB used twice: another user keeps ~2.5 GB there) -> arm M extraction, reverse order, on GPU3;
#  2. arm M0 fine-tune .done -> M0 extraction of held-out + inner-val videos on GPU1 (the M0 GPU);
#  3. all 50 arm M feature files present and no M extraction running -> long models: a3 runs on GPU2,
#     sed runs on GPU3 (GPU2 if GPU3 never freed). Exits when all three are launched.
cd "$(dirname "$0")/.."
LOG=results/phase2_watch.log
say() { echo "[$(date "+%F %T")] $*" >> $LOG; }
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1; }
g3=0; m0=0; lg=0; free3=0
say "watcher start"
while [ $((g3 + m0 + lg)) -lt 3 ]; do
  if [ $g3 = 0 ] && ! pgrep -f "exp19_queue.sh 3" >/dev/null && [ "$(mem 3)" -lt 6000 ]; then
    free3=$((free3 + 1))
    if [ $free3 -ge 2 ]; then
      if [ "$(ls results/M/fold0/untrimmed/*.npz 2>/dev/null | wc -l)" -lt 45 ]; then
        tmux new-session -d -s exp21_extM_rev "bash run_files/exp21_extract.sh M 3 rev"; say "GPU3 free: started M extraction rev"
      else say "GPU3 free: M extraction nearly done, not sharded"; fi
      g3=1
    fi
  fi
  if [ $m0 = 0 ] && [ -e results/M0/fold0/.done ] && ! pgrep -f "exp21_arm.sh M0" >/dev/null; then
    sleep 30
    if [ "$(mem 1)" -lt 500 ]; then
      tmux new-session -d -s exp21_extM0 "PATIENT_SET=eval bash run_files/exp21_extract.sh M0 1 fwd && bash run_files/exp21_eval.sh results/M0/fold0/untrimmed results/M0/fold0/splits.json"; say "M0 done: started M0 eval-set extraction on GPU1"; m0=1
    fi
  fi
  if [ $lg = 0 ] && [ "$(ls results/M/fold0/untrimmed/*.npz 2>/dev/null | wc -l)" -ge 50 ] \
     && ! pgrep -f "exp21_extract.sh M " >/dev/null; then
    mkdir -p results/M/fold0/long
    tmux new-session -d -s exp21_long_a3 "RUNS=\"a3 a3_causal\" bash run_files/exp21_long.sh M 2"
    G=2; [ $g3 = 1 ] && [ "$(mem 3)" -lt 6000 ] && G=3
    tmux new-session -d -s exp21_long_sed "RUNS=\"sed sed_causal\" bash run_files/exp21_long.sh M $G"
    say "features complete: started long models (a3 on GPU2, sed on GPU$G)"; lg=1
  fi
  sleep 60
done
say "watcher done"
