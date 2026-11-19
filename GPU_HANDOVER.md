# GPU handover EXP-19 -> EXP-21 (2026-11-19)

## (a) What runs (checked 11:36-11:50)
- GPU0: another user (10 GB, not ours). Never touched.
- GPU1: EXP-21 arm M, tmux `exp21_M` (`bash run_files/exp21_arm.sh M 1`), started 11:43.
- GPU2: EXP-19 queue lane 2 (tmux `exp19-lane2`, `run_files/exp19_queue.sh 2 "2026-11-20 08:00"`), job
  `AUG_none_fold1` (aug pilot, config none, fold 1), started ~11:04. Fine-tune at epoch 20/50 at 11:45
  (~102 s/epoch), then 1 fps features (--amp) + 2 MS-TCN heads + eval. ETA ~14:00-14:45 (a pilot config took 3.7 h).
- GPU3: EXP-19 queue lane 3 (tmux `exp19-lane3`, `run_files/exp19_queue.sh 3 "2026-11-20 08:00"`), job `S_fold2`
  (arm S, fold 2): fine-tune done 10:54, 4 Hz feature extraction 12/50 videos at 11:50 (~110 clips/s, ~4.5 min
  per video), then 2 temporal heads (~5 min each) + eval. ETA ~15:00-15:30.
- Lanes 0 and 1 (tmux `exp19-lane0/1`, GPUs 0 and 1, start time "2026-11-19 20:00") were idle (waiting for 20:00).

## How EXP-19 chains its jobs
`run_files/exp19_queue.sh <gpu> [deadline] [start]`: one lane per GPU loops over the pool
`S_fold2 AUG_none_fold1 S_fold3 AUG_none_fold2 S_fold4 AUG_none_fold3 AUG_none_fold4`; a lane claims a job by
`mkdir results/queue/<job>` (owner GPU in `<job>/gpu`). Before each claim `available()` checks that some pool job
has no claim dir (or is its own unfinished claim); if none, the lane logs `lane done` and exits. A running job is
never interrupted by this.

## (b) Hold (EXECUTED 2026-11-19 11:37, user request "quando finiscono i job metti in pausa EXP-19")
Created claim dirs for the 5 not-started jobs, owner `hold-exp21`, each with a `HOLD_EXP21` note:
`results/queue/{S_fold3,AUG_none_fold2,S_fold4,AUG_none_fold3,AUG_none_fold4}` (in ~/workspace/wt/short-long-context).
Effect: lanes 0 and 1 exited at once (`lane done` 11:37:17, they had no job). Lanes 2 and 3 keep running their
current job (`AUG_none_fold1`, `S_fold2`) untouched and exit when it finishes. No process was killed or signalled.
Paused at: arm S folds 3 and 4 not started (S folds 0-2 done/finishing); aug pilot config none folds 2, 3, 4 not
started (fold 0 done, fold 1 finishing). The `.done` markers of cv_finetune.py make every job resume cleanly.

Watcher (tmux `exp21_gpuwatch`, `run_files/exp21_gpuwatch.sh`, log `results/gpuwatch.log`): when lane 2 or 3 has
exited and its GPU is free (twice, 30 s apart) — or when EXP-21 arm M is done on GPU1 — it starts EXP-21 arm M0
in tmux `exp21_M0` on that GPU, then exits. The second EXP-19 GPU stays free for the next EXP-21 phases.

## (c) Resume EXP-19 (after EXP-21; run from a shell on ORSI-GPU2)
```bash
cd ~/workspace/wt/short-long-context
# 1. drop only the hold claims (never a dir with job.log, i.e. a job that ran)
for d in results/queue/*/; do [ -e "$d/HOLD_EXP21" ] && [ ! -e "$d/job.log" ] && rm -r "$d"; done
# 2. check the GPUs are free of EXP-21 (nvidia-smi), then relaunch the lanes (one per GPU you give back;
#    the old deadline 2026-11-20 08:00 is past by then: pass a new one or none)
for g in 2 3; do
  tmux new-session -d -s exp19-lane$g "cd ~/workspace/wt/short-long-context && export NUMPY_MADVISE_HUGEPAGE=0 && bash run_files/exp19_queue.sh $g >> results/queue/lane$g.log 2>&1"
done
```
Lanes 0/1 had start "2026-11-19 20:00": relaunch them the same way (`exp19_queue.sh 1 ...`) only if wanted;
lane 0 runs on GPU0, which another user occupies.
