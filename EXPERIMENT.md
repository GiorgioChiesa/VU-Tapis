# EXPERIMENT Esp6: phase decoding without flicker (Orsi, surgical phases)

Written 2026-11-13 12:40 CET, before any decoding run. Predictions are not edited afterwards.

Parent: container @ 73b22c4 (branch exp6-phase-decoding, worktree ~/workspace/wt/exp6-phase-decoding)
Data: orsi_tensors labels (`Label/*_all_labels.csv`, phase_id, 1 fps), 50 videos, F_base phase
posteriors already saved by train_hybrid.py in outputs/orsi/EXP2/F_base/seed{0,1}/fold{0..4}/val/*.npz
(CPU only, no GPU, no retraining for 6a/6b; <= 8 threads).

## Esp6a (diagnostics)

Hypothesis A1: the F_base phase error is mostly over-segmentation (2.30 predicted changes per true
change) rather than wrong segment identity.
Hypothesis A2 (oracle arm 2.3, phase acc 0.939 with ground-truth phases as input): not caused by
`fillna(0)` (checked: 0 NaN / unmapped phase ids in 393 141 rows) but by the training crop sampler of
train_hybrid.py: `start ~ U[0, n - crop]`, crop 4096 s, so seconds near the start and the end of each
video are almost never inside a training crop. Measured on fold 0 train videos: exposure / frequency
ratio Idle 0.13, Specimen_and_instrument_removal 0.26, Port_insertion 0.26, End_of_operation 0.42; oracle
recall of Idle 0.001 and Specimen removal 0.022 (49% of the oracle errors).
Command (A2 check, CPU, fold 0 seed 0 only, oracle arm):
  python tools/train_hybrid.py <2.3 args> --device cpu --crop-mode start   (reproduction)
  python tools/train_hybrid.py <2.3 args> --device cpu --crop-mode center  (fix)
Prediction A2: reproduction phase acc within 0.015 of the GPU run of fold 0 seed 0; centre crop
raises Idle recall above 0.5 and Specimen removal recall above 0.5, fold-0 phase acc +0.02 or more.
Stop rule A2: one run per mode; if the reproduction is off by more than 0.015, report and do not
conclude on the fix.

## Esp6b (decoding)

Hypothesis B: decoding the F_base posteriors with per-phase duration priors (and order) removes
flicker without hurting frame accuracy, and raises segmental F1@50 by >= 10 points.
Command: python tools/exp6_decode_cv.py --runs F_base/seed0 F_base/seed1 (see README in the commit).
Metric: segmental F1@50 (per video, mean over the 10 videos of the test fold; 10 paired runs = 5 folds x
2 seeds). Secondary: F1@10/25, edit, frame macro-F1, accuracy, Jaccard, boundary F1 +-5 s, ratio of
predicted to true segments, per-phase F1.
Pre-registered decision rule: a variant WINS vs P0 (raw argmax) if, in the 10 paired runs,
  F1@50 improves by >= 10 points in >= 8/10 runs, AND mean frame macro-F1 drops <= 0.01, AND mean
  accuracy drops <= 0.005, AND mean predicted/true segment ratio is within 0.8-1.3.
  Paired Wilcoxon vs P0 reported for every metric (two-sided, n = 10; min p = 0.002).
Hyperparameters tuned only on the out-of-fold posteriors of the 40 training videos of each outer fold
(tuning objective: mean per-video F1@50); priors for decoding a training video of inner fold j fitted on
the videos outside folds {k, j}. A fixed-default (untuned) version of every variant is also reported.
Prediction B (pooled F1@50 of F_base = 46.1 known; per-video values unknown before the run):
  P0-ma15:      F1@50 +5..+12, ratio ~1.5, passes F1 only partly -> rule FAILS (ratio > 1.3 likely).
  P1 (per-phase min duration + hysteresis, tuned): F1@50 +15..+25, acc change -0.005..+0.01,
     ratio 0.9-1.2 -> rule PASSES.
  P2 HSMM learned transitions (tuned): F1@50 +18..+28, acc +0.00..+0.02 -> PASSES (best variant).
  P2 strict order: similar F1@50 but acc may drop (videos off the modal order) -> 50% chance to fail on acc.
  P2 durations only: between P1 and P2-learned.
  P3 PELT + segment vote: F1@50 +12..+22 -> passes F1, ratio uncertain.
  Boundary F1 +-5 s: stays low (< 0.35) for every variant: raw recall at +-5 s is only 0.25, so most
  true boundaries are localised > 5 s off, which decoding cannot fix.
Stop rule: one pass of the grid below, no extra variants after seeing test-fold numbers; if a decoder
bug is found after the run, fix and re-run everything once, and say so in the note.
Budget: ~3 h wall clock, CPU only (<= 8 processes x 1 thread), no GPU hours.

Result (2026-11-13, commit 6f3d6b6, data: F_base posteriors exported from EXP2 seed0/seed1, 50 videos,
0 unmapped label rows; command run exactly as:
  OMP_NUM_THREADS=1 python tools/exp6_decode_cv.py --runs outputs/orsi/EXP6/F_base/seed0 \
      outputs/orsi/EXP6/F_base/seed1 --out outputs/orsi/EXP6/decode --workers 8
140 units, 25 min wall clock. Full tables: outputs/orsi/EXP6/decode/summary.{md,json}.)
Deviation: a smoke run on seed0 (4 default variants) showed that modal_order broke a median tie
(Right vs Left lateral dissection, both 0.5) by class id, i.e. against the 30/50 majority. Fixed
(tie broken by the mean position) before the full run; everything was run once after the fix.

Esp6b, with Idle, mean over 10 units (delta vs P0; runs with F1@50 gain >= 10; rule):
  P0_argmax           F1@50 52.9  edit 45.0  macroF1 0.764  acc 0.875  ratio 2.23  (pooled F1@50 46.1)
  P0_ma15             70.0 (+17.1) edit 67.4 macroF1 +0.001 acc +0.000 ratio 1.16  9/10  WIN
  P0_ma_tuned         77.2 (+24.3) edit 78.1 macroF1 -0.001 acc +0.000 ratio 0.91 10/10  WIN
  P1_default          78.9 (+26.0) edit 79.5 macroF1 +0.000 acc +0.002 ratio 0.89 10/10  WIN
  P1_tuned            78.8 (+25.9) edit 80.2 macroF1 -0.0105 acc -0.003 ratio 0.85 10/10 fail (macroF1)
  P1_perphase         76.7 (+23.8) edit 77.9 macroF1 -0.025 acc -0.010 ratio 0.85 10/10  fail
  P2_learned_default  76.9 (+24.0) edit 76.7 macroF1 +0.002 acc +0.001 ratio 0.94 10/10  WIN
  P2_learned_tuned    80.0 (+27.1) edit 80.4 macroF1 +0.002 acc +0.003 ratio 0.87 10/10  WIN
  P2_order_default    78.7 (+25.8) edit 80.3 macroF1 -0.025 acc -0.019 ratio 0.81 10/10  fail
  P2_order_tuned      78.8 (+25.9) edit 79.7 macroF1 -0.026 acc -0.019 ratio 0.81 10/10  fail
  P2_free_default     77.1 (+24.1) edit 76.6 macroF1 +0.001 acc +0.001 ratio 0.93 10/10  WIN
  P2_free_tuned       80.1 (+27.2) edit 80.1 macroF1 +0.000 acc +0.003 ratio 0.86 10/10  WIN
  P3_default          78.1 (+25.2) edit 78.2 macroF1 +0.003 acc +0.001 ratio 0.90 10/10  WIN
  P3_tuned            80.9 (+28.0) edit 80.7 macroF1 +0.002 acc +0.002 ratio 0.85 10/10  WIN
  Wilcoxon p for F1@50 = 0.002 (minimum, n = 10) for every variant. Boundary F1 +-5 s: 0.189 (P0) to
  0.268 (P3_tuned). Tuned values at the grid edge: P3 penalty 50 (10/10), moving-average window 121
  (10/10), HSMM temp 4 and dur_weight 2 (most units).
Esp6a, oracle arm 2.3 (fold 0, seed 0): GPU 0.902, CPU start-crop 0.905 (reproduced within 0.015),
  CPU centre-crop 0.912 (+0.007). Recall Idle 0.000 -> 0.991; Specimen_and_instrument_removal 0.000 ->
  0.000. Over all 100 GPU oracle videos (2 seeds) acc 0.936 / 0.940; error share Specimen 0.33,
  Idle 0.15-0.16, VU-anastomosis 0.11-0.12.

Audit follow-up (2026-11-13): the two seeds share the outer folds, so the 10 run x fold units are not
independent. Video-level re-analysis from the saved labels (no re-decoding; per-fold means match units/
exactly):
  python tools/exp6_decode_cv.py --runs outputs/orsi/EXP6/F_base/seed0 outputs/orsi/EXP6/F_base/seed1 \
      --out outputs/orsi/EXP6/decode --videolevel
Per video the metric is averaged over the 2 seeds; paired two-sided Wilcoxon vs P0_argmax over the 50
videos and over the 5 folds (smallest attainable p = 0.0625). F1@50 with Idle:
  variant        mean   median delta  improved/50  p (videos)  p (folds)  pooled F1@50
  P0_argmax      52.9   -             -            -           -          45.5
  P1_default     78.9   +27.0         50           1.8e-15     0.0625     78.2
  P2_free_tuned  80.1   +28.1         50           1.8e-15     0.0625     79.5
  P3_tuned       80.9   +28.4         50           1.8e-15     0.0625     80.4
Pooled = segment counts summed over all 50 videos of a seed, mean of the 2 seeds (the 46.1 quoted above
for P0 is the mean over units of the within-unit pooled value). Full tables:
outputs/orsi/EXP6/decode/summary_videolevel.{md,json}.
