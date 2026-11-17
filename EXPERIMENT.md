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
