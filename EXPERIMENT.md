# EXP-21: 4 fps multi-task fine-tune on the 8 target events + Mamba2 long context (fold 0)

Question: does a 4 fps multi-task fine-tune focused on the 8 target events (event head 8 targets +
background, phase head, TAPIS-teacher action and instrument heads) localise the target events better
than the EXP-19 S reference on the same fold 0?

Hypothesis: removing the 25 non-target event classes (masked, not background), gaussian targets at
4 Hz, balanced sampling around target events and the auxiliary phase/TAPIS-teacher losses give the
encoder sharper short-context evidence; a Mamba2 on the 4 Hz latent sequence (or its confidence-weighted
fusion with the encoder) recovers the long-context events.

Parent: exp/short-long-context (8a3f728), EXP-19 arm S fold 0 (encoder init).

Arms (same seed, same schedule, fold 0 only):
- M: events (9) + phases (15) + TAPIS actions (14) + TAPIS instruments (7).
- M0: events + phases only (no teacher heads). M - M0 = effect of the TAPIS teacher.
- Long-context models on M latents (L-A3 BiMamba2+MS-TCN, L-SED SEDMamba), fusion (encoder / long /
  confidence-weighted), online causal variant: later phases of this experiment.

Reference: EXP-19 S fold 0 existing predictions re-evaluated on the same new labels (evaluation only).

Command: see "Commands" below (filled as runs start; run exactly as written).

Metric: spotting AP per class (8 targets) and group mAP (short, long) at +-1, +-2, +-5 s on the fold 0
held-out videos (10), revised labels at fractional times; all 33 classes as a control where applicable.
Phases: accuracy, F1@50 with and without P1.

Prediction (written 2026-11-19, before any run):
- short group mAP@+-2 s (M encoder head, nms) >= reference + 0.03;
- long group: best of {encoder, Mamba2, fusion} mAP@+-2 s >= reference.

Stop rule: if neither group improves by >= 0.03 mAP@+-2 s over the reference, do not extend to the other
folds. Every choice (epoch, thresholds, fusion weight, long model) is made on fold 0 inner val only,
never on the fold 0 test videos.

Budget: fine-tune 8-15 GPU h per arm (measured estimate below), features 2-4 h per arm, long models
~1-1.5 h, evaluation 1-2 CPU h. GPU1 only for now.

Data version: labels v2026-11-19 / v2026-11-19_4hz (see "Labels"); frames data/orsi_tensors_4fps.

Result:

## Labels (plan step 0, built 2026-11-19 11:2x)
`data/orsi_labels_revised/v2026-11-19` (+ `_4hz`, with `row_4fps`), label-review commit cb42713, same params as
v2026-11-17 (`--skipped ignore --ignore-sec 3 --times`), from a snapshot of `viewer/revisione_etichette`
(1151 review lines, `_review_snapshot/`, BUILD.txt). Existing versions and original CSVs untouched.
- items reviewed 1067/3761 (v2026-11-17: 957); confirmed 395, changed 21, moved 622, changed+moved 8, skipped 21
  (ignore windows); builder: stamps matched 2953, unmatched 0, rows added 3.
- 4 Hz grid: 1 572 393 rows, 3974 events, 505 ignore rows, 0 events inside ignore windows (same as v2026-11-17).
Fold 0 (cv_finetune rule; inner val recomputed with choose_inner_val on v2026-11-19_4hz = same as EXP-19 S):
train 35, inner val RARP16 18 32 36 44, held out RARP35 13 12 40 20 19 49 41 08 07.
Target event instances (Grasp_cath, Clip_vas_def, SV_arteries, Metal_clip | Start_diss, Denonv, Transection, Stitch_bladder):
- train (35):      33  65 320 584 | 35 33 34 33
- inner val (5):    5   8  59  64 |  5  4  5  5
- test (10):        9  17  50 123 | 10  9 10 10   (identical counts on v2026-11-17; times moved by the review)

## Reference: EXP-19 S fold 0 re-evaluated (results/reference/<version>, run_files/exp21_reference.sh)
Eval: eval_untrimmed / eval_constraints, --fractional-gt, 4 Hz labels, ignore windows of the 1 fps version, 10 test videos.
Sanity check v2026-11-17: reproduces EXP-19 exactly (dilate3 nms short 0.532/0.722/0.848, caps_chain
0.551/0.738/0.871; mvit nms short 0.213/0.373/0.573, long 0.074/0.124/0.163; all-class mAP@5 identical).
Group mAP@1/@2/@5 on v2026-11-19:
| predictions | short | long |
|---|---|---|
| S MViT per clip, nms | 0.213 0.373 0.573 | 0.076 0.125 0.163 |
| S MS-TCN dilate3 seed 0, nms | 0.532 0.722 0.848 | 0.242 0.429 0.611 |
| S MS-TCN dilate3 seed 0, caps_chain | 0.551 0.738 0.871 | 0.229 0.428 0.631 |
| S MS-TCN dilate3 seed 1, nms | 0.529 0.811 0.890 | 0.188 0.297 0.591 |
| S MS-TCN dilate3 seed 1, caps_chain | 0.530 0.820 0.905 | 0.193 0.299 0.592 |
Per class AP@1/@2/@5, v2026-11-19 (MViT nms | MS-TCN s0 nms):
- Grasping_catheter_tip           0.270 0.498 0.654 | 0.314 0.330 0.740
- Clip_or_coagulate_vas_deferens  0.131 0.182 0.468 | 0.487 0.844 0.845
- Identification_clipping_SV_art  0.247 0.402 0.594 | 0.621 0.806 0.872
- Metal_clip_on_right_pedicle     0.204 0.409 0.574 | 0.707 0.907 0.936
- Start_dissection                0.050 0.159 0.166 | 0.317 0.454 0.768
- Incision_of_Denonvilliers       0.120 0.128 0.150 | 0.367 0.618 0.703
- Transection_of_the_urethra      0.010 0.011 0.128 | 0.091 0.252 0.531
- Stitch_in_bladder               0.122 0.203 0.208 | 0.193 0.391 0.441
Only Transection / Stitch (long) move with the new labels. Reference for the prediction: short = MS-TCN (mean of
seeds, nms) 0.767 mAP@2; per-clip encoder head 0.373. Long: 0.363 (mean of seeds, nms).

## Code (branch exp/focus8-multitask)
- `tapis/datasets/focus8.py`: target mapping (Orsi collapses), gaussian targets + non-target mask, phases (EXP-19 S
  local order), teacher per-second signals (as orsi_signals.py, no-region seconds masked) and window max, row
  sampling weights, soft CE, asymmetric BCE.
- `tools/train_focus8.py` + `configs/Orsi/FOCUS8/exp21_M.yaml`: fine-tune, resumable (last.pyth every epoch),
  best.pyth by inner-val frame mAP of the 8 targets (all rows within 3 s of a target + one row per 10 s).
  Deviation from the plan: a dedicated trainer instead of new cv_finetune.py flags (same fold rule, imported
  from cv_finetune.py); TAPIS train_net has no soft/masked/BCE losses. Temporal jitter not used (targets are exact).
- `tests/test_focus8.py`: 8 tests pass (incl. target counts == Orsi.collapse_event_dfs on 10 train videos,
  teacher == orsi_signals on RARP01); mutation check: removing the Hemolock collapse fails the count test.
- Smoke (2 train videos, 1 inner val, 48 clips, GPU1): shapes ok, losses finite.

## Timing
EXP-19 S fold 0 fine-tune: 50 epochs in 2 h 14 min (03:09-05:23, ~155 s/epoch, ~3500 clips/epoch); its 4 Hz
features of 50 videos took ~10.5 h at 41 clips/s (fp32; ~110 clips/s with --amp on fold 2).
EXP-21 M: 30 clips/s, epoch = 263 s train + 66 s inner val (7027 rows) -> 25 epochs ~2.4 h per arm; features
(50 videos, 4 Hz, --amp) ~4-5 h per arm.

## Commands (run exactly as written; seed 0)
- reference: `bash run_files/exp21_reference.sh v2026-11-17` and `... v2026-11-19` (CPU)
- arm M: tmux exp21_M: `bash run_files/exp21_arm.sh M 1` (started 2026-11-19 11:43, commit 2141846, GPU1)
- arm M0: tmux exp21_M0 via watcher `run_files/exp21_gpuwatch.sh` (= `bash run_files/exp21_arm.sh M0 <gpu>`)
