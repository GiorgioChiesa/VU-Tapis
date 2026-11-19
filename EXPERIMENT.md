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

Data version: see "Labels" below.

Result:
