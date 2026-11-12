# EXPERIMENT: BGR vs RGB input to the GraSP-trained TAPIS encoder (Orsi event spotting)

Hypothesis: The Orsi 1 fps tensors are stored BGR, while the GraSP-trained TAPIS encoder
(`LONG.pyth`, trained with ENDOVIS_DATASET.BGR=False) expects RGB. Feeding channel-flipped
(RGB) frames removes this input mismatch, so the frozen per-second features carry more
event information and the dilate3 MS-TCN head spots events better.

Parent: container @ 73b22c4 (branch exp/bgr-rgb-grasp-features, worktree ~/workspace/wt/bgr-rgb)

Command:
  PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
  RUN=/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis/outputs/orsi/LONG/control/totale
  ENC=/home/gchie/workspace/VU-Tapis/data/pretrained_models/train/LONG.pyth   (md5 54700c4e...)
  FD=/home/gchie/workspace/data/orsi_tensors
  1. Reproduction (no flip, one video RARP65):
     $PY tools/infer_untrimmed.py --run $RUN --ckpt checkpoint_best_steps.pyth --split all --patients RARP65 \
        --frame-dir $FD --encoder-ckpt $ENC --normalize --features --out $RUN/untrimmed/grasp_LONG_repro_bgr/all --gpu G
     compare with $RUN/untrimmed/grasp_LONG/all/RARP65.npz (features, probs_*).
  2. RGB features, 4 shards (as the original grasp_shard*.log), at most 2 GPUs at a time:
     $PY tools/infer_untrimmed.py --run $RUN --ckpt checkpoint_best_steps.pyth --split all --shard i/4 \
        --frame-dir $FD --encoder-ckpt $ENC --normalize --features --flip-channels \
        --out $RUN/untrimmed/grasp_LONG_rgb/all --gpu G
  3. Temporal heads, both feature sets, folds 0..4, seed 0, default args (as run_files/orsi_temporal.sh, no --wandb):
     $PY -W ignore tools/train_temporal.py --feat-dir <F> --target dilate --k 3 --cv-fold i/5 \
        --out /home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis/outputs/orsi/TEMPORAL_bgr_rgb/cv_grasp_dilate3_{bgr,rgb}/fold$i --gpu G
     F = $RUN/untrimmed/grasp_LONG/all (bgr) or $RUN/untrimmed/grasp_LONG_rgb/all (rgb)
  4. Per fold: eval_untrimmed.py --pred-dir <out>/val --frame-dir $FD ;
     eval_constraints.py --pred-dir <out>/val --frame-dir $FD --alpha 0.1 --phase-smooth 1 --gate-floor 0.2 ;
     summarize_temporal.py --root .../TEMPORAL_bgr_rgb  (no --wandb)

Metric: untrimmed spotting mAP@+-5 s (NMS 5 s), mean over 5 video-level CV folds; paired per-fold delta RGB - BGR.
  Secondary: mAP@+-1 s, F1@5 s, FA/h at best F1, phase accuracy, caps_chain mAP@5.

Prediction (written before any run): RGB - BGR mAP@5 mean delta = +0.015 (plausible range -0.01 .. +0.04);
  the MS-TCN partly compensates for the colour swap, so a small gain, positive in 3-4 of 5 folds.
  Re-run BGR baseline should land within +-0.01 of the reported 0.458 +- 0.016.
  Decision rule (agreed with the user): delta >= +0.02 on the mean AND positive in >= 4/5 folds -> adopt RGB;
  |delta| < 0.02 or negative -> close.

Stop rule:
  - Stop and report if the no-flip reproduction does not match the existing grasp_LONG features for RARP65
    (pass = np.allclose(new, old, rtol=1e-2, atol=1e-2) on fp16 features and max abs diff of probs < 1e-3).
  - Stop and report if the projected total time after the pilot exceeds 6 h.
  - Otherwise run steps 2-4 once and stop, whatever the result (no tuning, no extra seeds in this experiment).

Budget: about 2.5 h wall clock, about 3 GPU-hours, at most 2 GPU processes at once.

Result:
