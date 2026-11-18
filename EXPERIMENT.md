# EXP-19: short vs long context on revised labels

Hypothesis: for short events, a short-window encoder (16 frames at 4 fps = 4 s) followed by an
MS-TCN at 4 Hz, trained on labels revised to the quarter second, localises better than the
long-context model (16 frames at 1 fps = 16 s).

Parent: feat/label-review (98cfe60)

Arms (all evaluated on the revised ground truth v2026-11-17, fractional times, same 5-fold CV
seed 1234, ignore windows from the 1 fps revised labels for every arm):
- A0: existing cv_ft_dilate3 + caps_chain, no retraining.
- L: same pipeline retrained on revised 1 fps labels, ADD_IDLE 20, Idle gap 30 s.
- S: encoder fine-tuned on 4 fps clips, features and MS-TCN at 4 Hz, revised 4 Hz labels.

Classes:
- short: Grasping_catheter_tip, Clip_or_coagulate_vas_deferens_-_right,
  Identification_and_clipping_of_SV_arteries_-_right, Metal_clip_on_right_pedicle
- long: Start_dissection, Incision_of_Denonvilliers_fascia, Transection_of_the_urethra,
  Stitch_in_bladder

Command: see "Commands" below (A0 eval, arm L, arm S pilot).

Metric: spotting mAP@1 s and mAP@5 s (nms and caps_chain) on the short and long class groups,
per fold; Wilcoxon over 5 folds x 2 head seeds.

Prediction: on the short events S - L >= +0.05 mAP@1 with mAP@5 within -0.02; on the long events
L ~ A0 and S not better than L.

Stop rule: pilot gate on fold 0: if S does not beat L by >= 0.03 mAP@1 on the short events, stop
before the full S cross-validation. S is adopted for short events only if mAP@1 rises >= 0.03
(Wilcoxon p < 0.05) without losing more than 0.02 mAP@5.

Budget: L ~10-12 h on GPUs 2-3; S pilot ~8 h; S folds 1-4 ~25-30 h (only if the gate passes).

Data version: labels data/orsi_labels_revised/v2026-11-17 (+ _4hz), built from a snapshot of
1029 review lines (`_review_snapshot/`, BUILD.txt); frames data/orsi_tensors_4fps.

Result:

## EXP-19 augmentation ablation pilot (fold 0)

Hypothesis: part of the full ORSI_AUG set used by arm L (photometric RandAugment, erase, surgical
smoke/specular, temporal jitter) hurts or does not help spotting; a smaller subset matches or beats L.

Parent: exp/short-long-context (1f691fb), arm L.

Command: `bash run_files/exp19_aug_pilot.sh chain` (GPU 2, tmux session `aug-pilot`), configs in order
AUG_none, AUG_geo, AUG_geo_photo, AUG_geo_surg, AUG_geo_jit (last only if it starts before the deadline).
Every config is arm L (revised labels v2026-11-17, ADD_IDLE 20, IDLE_GAP_SEC 30, LONG.pyth init,
MS-TCN dilate3 heads seeds 0 and 1, caps_chain eval with the L EVAL args), fold 0 only, MAX_SAVE_VIDEO 0:
- none: ORSI_AUG.ENABLE False, jitter 0
- geo: flip 0.5, crop 1.0, rotation 0.5 only, jitter 0
- geo_photo: geo + COLOR_P 0.8 + BLUR_P 0.2, jitter 0
- geo_surg: geo + SMOKE_P 0.2 + SPECULAR_P 0.2, jitter 0
- geo_jit: geo + TEMPORAL_JITTER_ROWS 2
Reference: results/L fold 0 (full ORSI_AUG, jitter 2).

Metric: mAP@1 all classes, fold 0 test, caps_chain|a=0.1|s=1|f=0.2, mean over head seeds 0 and 1;
mAP@5; short/long group mAP@1/@5; inner-val metric = best val/steps_mAP of the encoder fine-tune
(W&B), the selection metric of checkpoint_best_steps.

Prediction: no config beats L fold 0 (0.111 mAP@1, 0.507 mAP@5) by >= 0.02 mAP@1; AUG_none is the
worst on inner-val (>= 0.02 below the 0.665 of L); geo-only configs fall within 0.02 of L on mAP@5.

Decision rule: winner = best mean mAP@1 among configs whose mAP@5 is within 0.02 of the best mAP@5;
it must beat L fold 0 by >= 0.02 mAP@1. If test and inner-val pick different configs, the inner-val
choice wins and is flagged.

Stop rule: stop after the four (five) configs on fold 0; no config starts after 2026-11-19 08:00; no
all-fold confirmation inside this pilot.

Budget: ~3.3 h per config on GPU 2, ~13-17 GPU h.

Result:
