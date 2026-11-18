#!/usr/bin/env bash
# EXP-19 GPU queue: one lane per GPU (one job at a time), each lane in its own tmux session:
#   bash run_files/exp19_queue.sh <gpu> [deadline]   e.g. exp19_queue.sh 0 "2026-11-19 08:00"
# Jobs, in priority order: the arm S pilot (S_fold0, once the 4 fps frame cache is complete), then
# the folds of A0aug and L. Arm S folds 1-4 are never queued here.
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
RUN=run_files/exp19_short_long.sh
FR4=/home/gchie/workspace/data/orsi_tensors_4fps
Q=results/queue
PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
mkdir -p $Q
log() { echo "[$(date "+%F %T")] lane gpu$GPU: $*"; }
log "start (deadline: ${DEADLINE:-none})"
# W&B run list (time, arm, fold, step, seed, url) rebuilt from the W&B API; never fails the lane
runs_tsv() { WANDB_SILENT=true timeout 600 $PY tools/wandb_runs_tsv.py >/dev/null 2>&1 || log "wandb_runs.tsv not updated"; }
runs_tsv

POOL=()
[ "$(ls $FR4/*/.done 2>/dev/null | wc -l)" -ge 50 ] && POOL+=(S_fold0)
for arm in A0aug L; do for i in 0 1 2 3 4; do POOL+=("${arm}_fold$i"); done; done

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
  if [ "$arm" = S ]; then GPUS=$GPU bash $RUN S0 2>&1 | tee "$Q/$1/job.log"
  else GPUS=$GPU FOLDS=$fold bash $RUN "$arm" 2>&1 | tee "$Q/$1/job.log"; fi
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

waiting=""
while true; do
  if past_deadline; then log "deadline $DEADLINE reached, no new jobs"; break; fi
  available || break
  busy=$(gpu_busy)
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
