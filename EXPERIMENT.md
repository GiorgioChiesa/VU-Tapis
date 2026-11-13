# Esp7: dedicated phase-only temporal model (Orsi)

Hypothesis: the phase segmentation of F_base (MS-TCN G2 dilate3, one 64-channel trunk shared by events
and phases, 2 stages) is limited by over-segmentation (2.3 predicted changes per true change, edit 45),
not by frame accuracy. A model trained for phases only, with more refinement stages, a T-MSE smoothing
loss and class-balanced CE, should raise the segmental metrics (edit, seg F1@50) with flat frame metrics.
  - A1: phase-only MS-TCN, 2 stages, F_base phase loss (CE + 0.15 T-MSE, unweighted): isolates
        "phase-only vs shared head".
  - A2: phase-only MS-TCN, 4 stages, CE weighted by inverse sqrt train-fold frequency + 0.15 T-MSE.
  - A3: bidirectional Mamba2 encoder (pure PyTorch, vendored) as the first stage + 3 MS-TCN refinement
        stages, A2 losses.
Parent: container (73b22c4)
Command: bash run_files/exp7_phase_model.sh grid ; python tools/summarize_exp7.py
         (per run: tools/train_phase.py --arm {A1,A2,A3} --seed {0,1} --cv-fold {0..4}/5
          --feat-dir outputs/orsi/LONG/cv_ft/fold{F}/untrimmed/checkpoint_best_steps/all)
Data: per-fold fine-tuned encoder features (same as F_base), cv_split of train_hybrid.py (Random(1234),
      5 video-level folds), seeds 0 and 1 -> 10 paired runs per arm. Labels: orsi_tensors Label CSVs.
Model selection: inner validation split = 20 % of the training videos of the fold (Random(4321 + fold)),
      checkpoint every 5 epochs, selected by inner per-video seg F1@50 (phases); the model is then refit
      on all training videos of the fold with the same seed and schedule and stopped at the selected
      epoch. The test fold is never used for selection.
Metric: PRIMARY seg F1@50 of the argmax phases, per video, mean over the test videos of a run.
      Secondary: seg F1@10/25, edit, frame macro-F1 (per video, classes in GT or prediction), accuracy,
      Jaccard, boundary F1 +/-5 s, predicted/true segment ratio, per-phase F1 (pooled per run).
      F_base is re-evaluated with the same code from its saved `probs_phases`.
Decision rule (pre-registered): an arm is adopted if seg F1@50 >= F_base + 5 points in >= 8/10 paired
      runs AND mean frame macro-F1 >= F_base - 0.01.
Prediction (written before any run):
  - A1: seg F1@50 = F_base + 0 to +4 points (per-video mean); macro-F1 within +/-0.01; rule NOT met.
  - A2: seg F1@50 = F_base + 6 to +12 points; edit +8 to +15; pred/true segment ratio 1.3-1.7;
        macro-F1 between -0.01 and +0.02; rule met in 8-10/10 runs.
  - A3: same as A2 within +/-3 points of F1@50 (Orsi videos are long and phases are coarse; the
        MS-TCN refinement stages, not the encoder, drive the gain); rule outcome same as A2.
Stop rule: one grid (3 arms x 10 runs), no hyper-parameter search on the test folds. If neither A2 nor
      A3 meets the rule, close the "deeper phase model" branch and hand the posteriors to Esp6 decoding.
      A3 is dropped if its CPU smoke test exceeds 3x the A2 time per epoch.
Budget: CPU only (user holds 2 GPUs), <= 8 threads, <= 8 h wall clock for the grid; 0 GPU hours.
Result: (raw, per-video mean over 10 runs; summary.json, commit of this result)
  F_base   F1@50 52.9 edit 45.0 macroF1 0.764 acc 0.875 seg_ratio 2.23 pooled F1@50 46.1
  A1       F1@50 50.8 edit 42.1 macroF1 0.767 acc 0.880 seg_ratio 2.40 pooled F1@50 43.8 rule 2/10 -> not adopted
  A2       F1@50 72.7 edit 71.7 macroF1 0.768 acc 0.877 seg_ratio 1.12 pooled F1@50 70.2 rule 10/10 -> adopted
  A3       F1@50 75.5 edit 75.7 macroF1 0.760 acc 0.870 seg_ratio 0.98 pooled F1@50 74.2 rule 10/10 -> adopted
  + Esp6 P1_default (5029200): F_base 78.9 / A1 79.3 / A2 79.5 / A3 78.8 (F1@50); vs F_base+P1 video-level
    Wilcoxon (n=50, seed-averaged): A1 +0.4 p 0.52, A2 +0.6 p 0.41, A3 -0.1 p 0.46.
  Video level vs F_base: A2 +19.7 (48/50, p 4e-14), A3 +22.5 (47/50, p 2e-13), A1 -2.1 (25/50, p 0.42).
  Cost: A1 22.8 min/run CPU, A2 38.5 min/run CPU (2 threads), A3 12.6 min/run on GPU 0 (deviation: 2.1 GPU h).
