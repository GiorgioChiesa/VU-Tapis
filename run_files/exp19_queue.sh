#!/usr/bin/env bash
# EXP-19 GPU queue: one lane per GPU (one job at a time), each lane in its own tmux session:
#   bash run_files/exp19_queue.sh <gpu> [deadline] [start]   e.g. exp19_queue.sh 0 "2026-11-20 08:00" "2026-11-19 20:00"
# The lane claims nothing before the optional start time, nor while the optional AFTER_PID process lives
# (e.g. a job of another launcher on the same GPU, whose GPU can look free between two of its steps).
# Jobs, in priority order (JOBS="..." overrides): arm S folds 2-4 interleaved with the AUG_none folds 1-4 of
# the augmentation pilot (run_files/exp19_aug_pilot.sh, config none). Job names: <arm>_fold<i>, where arm is
# S, L, A0aug (run_files/exp19_short_long.sh) or AUG_<config> (run_files/exp19_aug_pilot.sh).
# Every job gets EXTRA_SET="$JOB_EXTRA_SET" (default "MAX_SAVE_VIDEO 1": one saved reference video per run) and
# INFER_ARGS="$JOB_INFER_ARGS" (default "--amp": fp16 feature extraction).
# A GPU is never shared: a lane claims a job only when its GPU has no compute process at all (ours
# or another user's) and almost no memory in use; otherwise it sleeps and rechecks, so the job goes
# to another free lane. cv_finetune.py repeats the same check before every step.
# Jobs are claimed with one directory per job (results/queue/<job>, owner GPU inside), so lanes
# never run the same job; a relaunched lane resumes its own unfinished claims first and the .done
# markers of cv_finetune.py skip finished steps. A job that exits non-zero or reports a FAILED fold
# is marked `failed` and keeps its claim (no retry): inspect it, then remove results/queue/<job> to
# requeue it. After the optional deadline the lane takes no new job (the running one finishes).
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${1:?usage: $0 <gpu> [deadline]}
DEADLINE=${2:-}
START=${3:-}
AFTER_PID=${AFTER_PID:-}
export EXTRA_SET=${JOB_EXTRA_SET:-MAX_SAVE_VIDEO 1}
export INFER_ARGS=${JOB_INFER_ARGS:---amp}
RUN=run_files/exp19_short_long.sh
AUG=run_files/exp19_aug_pilot.sh
Q=results/queue
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
mkdir -p $Q
log() { echo "[$(date "+%F %T")] lane gpu$GPU: $*"; }
log "start (start: ${START:-now}, deadline: ${DEADLINE:-none}, after pid: ${AFTER_PID:-none}, EXTRA_SET: $EXTRA_SET, INFER_ARGS: $INFER_ARGS)"
# W&B run list (time, arm, fold, step, seed, url) rebuilt from the W&B API; never fails the lane
runs_tsv() { WANDB_SILENT=true timeout 600 $PY tools/wandb_runs_tsv.py >/dev/null 2>&1 || log "wandb_runs.tsv not updated"; }
runs_tsv

read -r -a POOL <<< "${JOBS:-S_fold2 AUG_none_fold1 S_fold3 AUG_none_fold2 S_fold4 AUG_none_fold3 AUG_none_fold4}"
log "jobs: ${POOL[*]}"

owner() { cat "$Q/$1/gpu" 2>/dev/null; }
mine() { [ "$(owner "$1")" = "$GPU" ] && [ ! -e "$Q/$1/finished" ] && [ ! -e "$Q/$1/failed" ]; }
claim() {  # claim job $1 for this lane (or keep our own claim from a previous launch)
  if mkdir "$Q/$1" 2>/dev/null; then echo "$GPU" > "$Q/$1/gpu"; return 0; fi
  mine "$1"
}
available() { for j in "${POOL[@]}"; do if [ ! -e "$Q/$j" ] || mine "$j"; then return 0; fi; done; return 1; }
gpu_busy() {  # prints what occupies this GPU (compute processes with owner, or memory in use); empty = free
  local pid used out=""
  for pid in $(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits); do
    out+="pid $pid ($(ps -o user= -p "$pid" 2>/dev/null || echo unknown user)) "
  done
  used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-0}" -ge 1500 ] && out+="${used} MiB in use"
  echo -n "$out"
}
run_job() {  # $1 = <arm>_fold<i>
  local arm=${1%_fold*} fold=${1##*_fold} rc
  log "start $1"
  case $arm in
    AUG_*) GPUS=$GPU FOLD=$fold bash $AUG "${arm#AUG_}" 2>&1 | tee "$Q/$1/job.log" ;;
    S) if [ "$fold" = 0 ]; then GPUS=$GPU bash $RUN S0 2>&1 | tee "$Q/$1/job.log"
       else GPUS=$GPU FOLDS=$fold bash $RUN S 2>&1 | tee "$Q/$1/job.log"; fi ;;
    *) GPUS=$GPU FOLDS=$fold bash $RUN "$arm" 2>&1 | tee "$Q/$1/job.log" ;;
  esac
  rc=$?
  if [ "$rc" -ne 0 ] || grep -q "FAILED" "$Q/$1/job.log"; then
    touch "$Q/$1/failed"
    log "FAILED $1 (exit $rc): see $Q/$1/job.log; claim kept, no retry (remove $Q/$1 to requeue)"
  else
    touch "$Q/$1/finished"; log "end $1 (ok)"
  fi
  runs_tsv
}
past_deadline() { [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$(date -d "$DEADLINE" +%s)" ]; }
before_start() { [ -n "$START" ] && [ "$(date +%s)" -lt "$(date -d "$START" +%s)" ]; }

waiting=""
while true; do
  if past_deadline; then log "deadline $DEADLINE reached, no new jobs"; break; fi
  available || break
  if before_start; then sleep 60; continue; fi
  if [ -n "$AFTER_PID" ] && kill -0 "$AFTER_PID" 2>/dev/null; then
    [ "$waiting" != "pid" ] && log "waiting for pid $AFTER_PID to exit"
    waiting=pid; sleep 120; continue
  fi
  busy=$(gpu_busy)
  # the GPU must look free twice, 30 s apart: a running job is briefly off the GPU between its steps
  [ -z "$busy" ] && { sleep 30; busy=$(gpu_busy); }
  if [ -n "$busy" ]; then
    [ "$busy" != "$waiting" ] && log "GPU busy, not claiming: $busy"
    waiting=$busy; sleep 120; continue
  fi
  waiting=""
  job=""
  for j in "${POOL[@]}"; do if claim "$j"; then job=$j; break; fi; done
  [ -z "$job" ] && continue
  run_job "$job"
done
runs_tsv
log "lane done"
