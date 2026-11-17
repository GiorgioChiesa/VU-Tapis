#!/usr/bin/env bash
# EXP-19 GPU queue: one lane per GPU (one job at a time), each lane in its own tmux session:
#   bash run_files/exp19_queue.sh <gpu> [deadline]   e.g. exp19_queue.sh 0 "2026-11-18 08:00"
# A lane takes fold jobs of arms L and A0aug from a shared pool (GPUs 0-1 prefer A0aug, GPUs 2-3
# prefer L). GPU 3 runs the arm S pilot (fold 0) as soon as the 4 fps frame cache is complete
# (one .done per video, 50 videos), at its next free slot. Arm S folds 1-4 are never queued here.
# Jobs are claimed with one directory per job (results/queue/<arm>_fold<i>, owner GPU inside), so
# lanes never run the same fold; a relaunched lane resumes its own claims first and the .done
# markers of cv_finetune.py skip finished steps. After the optional deadline the lane takes no new
# job (the running one finishes). cv_finetune.py also waits for the GPU to be free before each step.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${1:?usage: $0 <gpu> [deadline]}
DEADLINE=${2:-}
RUN=run_files/exp19_short_long.sh
FR4=/home/gchie/workspace/data/orsi_tensors_4fps
Q=results/queue
mkdir -p $Q
log() { echo "[$(date "+%F %T")] lane gpu$GPU: $*"; }

case "$GPU" in
  0|1) ARMS="A0aug L" ;;
  *) ARMS="L A0aug" ;;
esac
POOL=()
for arm in $ARMS; do for i in 0 1 2 3 4; do POOL+=("${arm}_fold$i"); done; done

owner() { cat "$Q/$1/gpu" 2>/dev/null; }
claim() {  # claim job $1 for this lane (or keep our own claim from a previous launch)
  if mkdir "$Q/$1" 2>/dev/null; then echo "$GPU" > "$Q/$1/gpu"; return 0; fi
  [ "$(owner "$1")" = "$GPU" ] && [ ! -e "$Q/$1/finished" ]
}
run_job() {  # $1 = <arm>_fold<i>
  local arm=${1%_fold*} fold=${1##*_fold}
  nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader
  log "start $1"
  if [ "$arm" = S ]; then bash $RUN S0; else GPUS=$GPU FOLDS=$fold bash $RUN "$arm"; fi
  log "end $1 (exit $?)"; touch "$Q/$1/finished"
}
past_deadline() { [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$(date -d "$DEADLINE" +%s)" ]; }

while true; do
  if past_deadline; then log "deadline $DEADLINE reached, no new jobs"; break; fi
  job=""
  if [ "$GPU" = 3 ] && [ "$(ls $FR4/*/.done 2>/dev/null | wc -l)" -ge 50 ] && claim S_fold0; then job=S_fold0; fi
  if [ -z "$job" ]; then for j in "${POOL[@]}"; do if claim "$j"; then job=$j; break; fi; done; fi
  if [ -z "$job" ]; then
    # GPU 3: nothing left in the pool, but the S pilot may still be waiting for the frame cache
    if [ "$GPU" = 3 ] && [ ! -e $Q/S_fold0 ]; then sleep 120; continue; fi
    break
  fi
  run_job "$job"
done
log "lane done"
