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

## Mamba env and SEDMamba (plan step 5 prep)
- venv `~/workspace/envs/mamba` (`--system-site-packages` over Vu-Tapis; torch stays the base 2.6.0+cu124):
  `causal-conv1d==1.5.0.post8`, `mamba-ssm==2.2.4` (prebuilt cu12/torch2.6/cp313 wheels, no compile; build script
  `~/workspace/envs/build_mamba.sh`, constraint torch==2.6.0 because unpinned latest mamba-ssm pulls a cu13 torch).
  Import and CUDA forward OK on GPU1; `Mamba` and `Mamba2` blocks are causal (perturbing t >= 200 leaves outputs
  t < 200 bit-identical).
- SEDMamba cloned at `~/workspace/repos/SEDMamba` (a65417d, Apache-2.0). `baseline/SEDMamba.py`: 3 stages, each
  1x1 bottleneck -> FCTF (conv k3 dilation 2/4/8 chained, `padding=dilation`, i.e. symmetric) -> concat + 1x1 ->
  one unidirectional `mamba_ssm.Mamba` (v1, d_state 16, d_conv 4, expand 2) -> 1x1. The Mamba part is causal, the
  FCTF convs are NOT: measured, an input change at step 300 changes outputs from step 280. Causal variant for the
  online arm: left padding 2*dilation (padding=0 + F.pad(x, (2d, 0))). No bidirectional scan in the official code.

## Arm M fine-tune: summary (inner val only; raw numbers)
Ran 2026-11-19 11:43-14:00 (2 h 18 min, 25 epochs, exit 0, `.done`), commit 2141846, seed 0, GPU1, labels v2026-11-19_4hz.
best.pyth = epoch 13 (selected by inner-val frame mAP of the 8 targets, 7027 rows of RARP16 18 32 36 44; held-out
videos never loaded). Note: epoch 23 is the second best (0.265), the selection rule keeps epoch 13.
| epoch | frame mAP 8 | short | long | phase acc |
|---|---|---|---|---|
| 13 (best) | 0.270 | 0.394 | 0.146 | 0.489 |
| 23 | 0.265 | 0.406 | 0.125 | 0.532 |
| 24 (last) | 0.250 | 0.371 | 0.130 | 0.536 |
Inner-val frame AP per class at epoch 13: Grasping_catheter_tip 0.478, Clip_vas_def 0.302, SV_arteries 0.394,
Metal_clip 0.401 | Start_dissection 0.044, Denonvilliers 0.310, Transection 0.024, Stitch_in_bladder 0.208.
Curve: frame mAP 0.18-0.27 across epochs (noisy, 5 videos); train loss 4.42 -> 3.61.

## Phase 2 (features, encoder-head evaluation, long-context models): commands (run exactly as written)
- M features (4 Hz, 16-frame centred clips, `--amp`, all heads + 1536-d fp16 latents, resumable per video, atomic
  writes): tmux exp21_extM_fwd `bash run_files/exp21_extract.sh M 2 fwd` (2026-11-19 14:20, commit 79687fc, GPU2;
  order: held out, inner val, train); second shard `bash run_files/exp21_extract.sh M 3 rev` started by the watcher
  `run_files/exp21_phase2_watch.sh` (tmux exp21_p2watch, log results/phase2_watch.log) when GPU3 is free. 113 clips/s,
  ~5.5 min per video. Output results/M/fold0/untrimmed/RARPxx.npz (probs_steps 9, probs_phases 15,
  probs_tapis_actions 14, probs_tapis_instruments 7, features 1536).
- Evaluation: `bash run_files/exp21_eval.sh <pred dir>` -> `<pred dir>/eval/{inner_val,held_out}.json`
  (tools/eval_focus8.py: target spotting AP at +-1/2/5 s, decoders nms / caps / caps_chain / phase_caps_chain
  (alpha 0.1, phase smoothing 61 s, gate floor 0.2), priors from the 35 training videos (target events only),
  per-video bootstrap 95% CI (1000 resamples); phases row acc / macro F1, F1@50 per video at 1 Hz, P0 and P1 =
  tools/phase_decode.py decode_hysteresis defaults, copied from exp6-phase-decoding 6f3d6b6).
  Evaluator check: on EXP-19 S MS-TCN dilate3 seed 0 (target columns of the 33-class output) it reproduces the
  reference exactly for nms (short 0.532/0.722/0.848, long 0.242/0.429/0.611, all per-class AP equal) and for
  caps_chain short (0.551/0.738/0.871); caps_chain long 0.231/0.433/0.627 vs 0.229/0.428/0.631 (chain priors
  here use target events only). Test GT counts 9 17 50 123 | 10 9 10 10, as above.
- Long models: `bash run_files/exp21_long.sh M <gpu>` (tools/train_long_mamba.py, mamba venv; started by the
  watcher when the 50 feature files exist: a3 + a3_causal on GPU2, sed + sed_causal on GPU3), seeds 0 and 1,
  crop 16384 rows, 100 epochs, AdamW 5e-4 cosine, feature dropout 0.5, events = Idle + 4 long targets dilated
  +-1 s (weighted CE, inverse sqrt frequency), phases CE, + 0.15 T-MSE on both heads; selection = inner-val long
  mAP (nms) averaged over +-1/2/5 s, every 5 epochs.
  L-A3 = BiMamba2Encoder (d_model 64, 4 layers, d_state 32, headdim 16; tapis/models/mamba2_minimal.py copied from
  exp7-phase-model, + `bidirectional=False`) + 2 MS-TCN stages (64 ch, 10 layers); causal = forward scan + left
  padded MS-TCN. L-SED = official SEDMamba MultiStageModel (3 BMSS blocks, G = 64; classifier -> 5 + 15 outputs);
  causal = FCTF convs left padded. Causality checked: perturbing inputs from step 1500 leaves outputs < 1500
  bit-identical for both causal variants (offline SED already changes only from step 1483: its scan is causal).
  Smoke test (2 videos, 2 epochs) ok for the 4 variants.

## Reference re-evaluated with tools/eval_focus8.py (test, 10 videos; 95% CI = per-video bootstrap, 1000)
| predictions | decoder | short mAP@1 / @2 / @5 [95% CI] | long mAP@1 / @2 / @5 [95% CI] |
|---|---|---|---|
| S MViT per clip | nms | 0.213 [0.16, 0.34] / 0.373 [0.26, 0.52] / 0.573 [0.42, 0.73] | 0.076 [0.04, 0.17] / 0.125 [0.09, 0.23] / 0.163 [0.12, 0.27] |
| S MViT per clip | caps_chain | 0.223 [0.16, 0.35] / 0.386 [0.26, 0.53] / 0.581 [0.44, 0.75] | 0.079 [0.03, 0.20] / 0.132 [0.08, 0.26] / 0.170 [0.12, 0.30] |
| S MViT per clip | phase_caps_chain | 0.206 [0.15, 0.34] / 0.350 [0.24, 0.51] / 0.497 [0.37, 0.66] | 0.031 [0.01, 0.11] / 0.073 [0.04, 0.16] / 0.085 [0.06, 0.17] |
| S MS-TCN dilate3 s0 | nms | 0.532 [0.41, 0.69] / 0.722 [0.60, 0.85] / 0.848 [0.74, 0.93] | 0.242 [0.16, 0.39] / 0.429 [0.33, 0.63] / 0.611 [0.51, 0.79] |
| S MS-TCN dilate3 s0 | caps_chain | 0.551 [0.42, 0.72] / 0.738 [0.61, 0.87] / 0.871 [0.75, 0.96] | 0.231 [0.15, 0.39] / 0.433 [0.33, 0.65] / 0.627 [0.52, 0.81] |
| S MS-TCN dilate3 s0 | phase_caps_chain | 0.547 [0.41, 0.71] / 0.736 [0.61, 0.87] / 0.868 [0.75, 0.95] | 0.238 [0.16, 0.39] / 0.424 [0.33, 0.63] / 0.612 [0.52, 0.78] |
| S MS-TCN dilate3 s1 | nms | 0.529 [0.45, 0.62] / 0.811 [0.67, 0.92] / 0.890 [0.80, 0.95] | 0.188 [0.09, 0.39] / 0.297 [0.18, 0.52] / 0.591 [0.46, 0.78] |
| S MS-TCN dilate3 s1 | caps_chain | 0.530 [0.45, 0.62] / 0.820 [0.68, 0.93] / 0.905 [0.82, 0.96] | 0.193 [0.09, 0.40] / 0.299 [0.18, 0.52] / 0.592 [0.45, 0.78] |
| S MS-TCN dilate3 s1 | phase_caps_chain | 0.504 [0.42, 0.61] / 0.812 [0.67, 0.93] / 0.895 [0.81, 0.95] | 0.192 [0.08, 0.41] / 0.366 [0.25, 0.56] / 0.558 [0.43, 0.77] |

| predictions | row acc | row macro F1 | F1@50 P0 | F1@50 P1 |
|---|---|---|---|---|
| S MViT per clip | 0.512 | 0.401 | 0.2 | 13.9 |
| S MS-TCN dilate3 s0 | 0.835 | 0.713 | 41.3 | 71.3 |
| S MS-TCN dilate3 s1 | 0.832 | 0.710 | 40.3 | 70.4 |
(caps = caps_chain here up to 0.01; full numbers in results/reference/v2026-11-19/*_focus8/eval_test.json.)

## Arm M encoder heads (per-clip, 4 Hz, best.pyth epoch 13): raw results
Decoder chosen on inner val per group (mAP@2): short = caps (0.521 vs nms 0.497, caps_chain 0.501), long = nms (0.092 vs
caps 0.075). Selected on test: short caps 0.324 / 0.422 / 0.494, long nms 0.111 / 0.144 / 0.160. Inner val (5 videos):
| predictions | decoder | short mAP@1 / @2 / @5 [95% CI] | long mAP@1 / @2 / @5 [95% CI] |
|---|---|---|---|
| M enc inner val | nms | 0.410 [0.26, 0.58] / 0.497 [0.31, 0.72] / 0.542 [0.38, 0.75] | 0.042 [0.00, 0.22] / 0.092 [0.05, 0.25] / 0.100 [0.07, 0.25] |
| M enc inner val | caps | 0.418 [0.28, 0.57] / 0.521 [0.34, 0.70] / 0.583 [0.43, 0.75] | 0.038 [0.00, 0.21] / 0.075 [0.03, 0.24] / 0.096 [0.08, 0.24] |
| M enc inner val | caps_chain | 0.398 [0.26, 0.57] / 0.501 [0.31, 0.70] / 0.563 [0.41, 0.74] | 0.038 [0.00, 0.21] / 0.075 [0.03, 0.24] / 0.096 [0.08, 0.24] |
| M enc inner val | phase_caps_chain | 0.312 [0.20, 0.46] / 0.389 [0.24, 0.57] / 0.443 [0.36, 0.63] | 0.027 [0.00, 0.15] / 0.047 [0.02, 0.17] / 0.070 [0.03, 0.20] |

| predictions | row acc | row macro F1 | F1@50 P0 | F1@50 P1 |
|---|---|---|---|---|
| M enc inner val | 0.433 | 0.237 | 0.2 | 18.0 |

Test (10 held-out videos), with the reference:
| predictions | decoder | short mAP@1 / @2 / @5 [95% CI] | long mAP@1 / @2 / @5 [95% CI] |
|---|---|---|---|
| M enc test | nms | 0.321 [0.21, 0.45] / 0.419 [0.30, 0.55] / 0.468 [0.36, 0.59] | 0.111 [0.06, 0.21] / 0.144 [0.11, 0.23] / 0.160 [0.13, 0.25] |
| M enc test | caps | 0.324 [0.21, 0.46] / 0.422 [0.30, 0.56] / 0.494 [0.38, 0.62] | 0.125 [0.06, 0.23] / 0.183 [0.14, 0.27] / 0.216 [0.16, 0.31] |
| S MViT per clip | nms | 0.213 [0.16, 0.34] / 0.373 [0.26, 0.52] / 0.573 [0.42, 0.73] | 0.076 [0.04, 0.17] / 0.125 [0.09, 0.23] / 0.163 [0.12, 0.27] |
| S MViT per clip | caps | 0.220 [0.16, 0.35] / 0.382 [0.26, 0.52] / 0.576 [0.43, 0.74] | 0.080 [0.03, 0.20] / 0.125 [0.08, 0.25] / 0.162 [0.11, 0.28] |
| S MS-TCN dilate3 s0 | nms | 0.532 [0.41, 0.69] / 0.722 [0.60, 0.85] / 0.848 [0.74, 0.93] | 0.242 [0.16, 0.39] / 0.429 [0.33, 0.63] / 0.611 [0.51, 0.79] |
| S MS-TCN dilate3 s0 | caps | 0.551 [0.42, 0.72] / 0.738 [0.61, 0.87] / 0.871 [0.75, 0.96] | 0.231 [0.15, 0.39] / 0.433 [0.33, 0.65] / 0.627 [0.52, 0.81] |

| predictions | decoder | Grasp | ClipVas | SVart | MetalClip | StartDiss | Denonv | Transect | StitchBl |  (AP@1/@2/@5)
|---|---|---|---|---|---|---|---|---|---|
| M enc test | nms | 0.16 0.29 0.36 | 0.36 0.37 0.39 | 0.40 0.49 0.53 | 0.36 0.52 0.58 | 0.05 0.10 0.10 | 0.10 0.18 0.23 | 0.01 0.01 0.02 | 0.28 0.29 0.30 |
| M enc test | caps | 0.18 0.31 0.48 | 0.36 0.37 0.39 | 0.40 0.49 0.53 | 0.36 0.52 0.58 | 0.08 0.17 0.17 | 0.12 0.26 0.37 | 0.01 0.01 0.02 | 0.29 0.30 0.31 |
| S MViT per clip | nms | 0.27 0.50 0.65 | 0.13 0.18 0.47 | 0.25 0.40 0.59 | 0.20 0.41 0.57 | 0.05 0.16 0.17 | 0.12 0.13 0.15 | 0.01 0.01 0.13 | 0.12 0.20 0.21 |
| S MViT per clip | caps | 0.29 0.52 0.69 | 0.14 0.19 0.45 | 0.25 0.40 0.59 | 0.20 0.41 0.57 | 0.05 0.15 0.15 | 0.11 0.12 0.14 | 0.01 0.01 0.13 | 0.15 0.22 0.23 |
| S MS-TCN dilate3 s0 | nms | 0.31 0.33 0.74 | 0.49 0.84 0.85 | 0.62 0.81 0.87 | 0.71 0.91 0.94 | 0.32 0.45 0.77 | 0.37 0.62 0.70 | 0.09 0.25 0.53 | 0.19 0.39 0.44 |
| S MS-TCN dilate3 s0 | caps | 0.39 0.40 0.84 | 0.49 0.84 0.84 | 0.62 0.81 0.87 | 0.71 0.91 0.94 | 0.30 0.43 0.77 | 0.34 0.64 0.73 | 0.09 0.27 0.57 | 0.19 0.39 0.43 |

| predictions | row acc | row macro F1 | F1@50 P0 | F1@50 P1 |
|---|---|---|---|---|
| M enc test | 0.452 | 0.235 | 0.1 | 23.2 |
| S MViT per clip | 0.512 | 0.401 | 0.2 | 13.9 |
| S MS-TCN dilate3 s0 | 0.835 | 0.713 | 41.3 | 71.3 |

## Arm M0 fine-tune: summary (inner val only; raw numbers)
Ran 2026-11-19 14:02-16:19 (25 epochs, exit 0), commit fb694dd, seed 0, GPU1, same splits as M. best.pyth = epoch 24
(last): inner-val frame mAP 0.259, short 0.369, long 0.148, phase acc 0.540 (M best: 0.270 / 0.394 / 0.146 / 0.489).
Curve 0.17-0.26. M0 features (held out + inner val only): `PATIENT_SET=eval bash run_files/exp21_extract.sh M0 1 fwd`
then `bash run_files/exp21_eval.sh results/M0/fold0/untrimmed results/M0/fold0/splits.json` (watcher, 16:20, GPU1).

## Arm M0 encoder heads (best.pyth epoch 24): raw results, M vs M0
Inner val:
| predictions | decoder | short mAP@1 / @2 / @5 [95% CI] | long mAP@1 / @2 / @5 [95% CI] |
|---|---|---|---|
| M0 enc inner val | nms | 0.384 [0.23, 0.55] / 0.473 [0.28, 0.66] / 0.533 [0.39, 0.71] | 0.062 [0.00, 0.28] / 0.096 [0.04, 0.31] / 0.114 [0.07, 0.33] |
| M0 enc inner val | caps | 0.395 [0.25, 0.55] / 0.483 [0.31, 0.66] / 0.547 [0.41, 0.71] | 0.061 [0.00, 0.31] / 0.092 [0.03, 0.33] / 0.109 [0.05, 0.35] |
| M0 enc inner val | caps_chain | 0.395 [0.25, 0.55] / 0.483 [0.31, 0.66] / 0.547 [0.41, 0.71] | 0.061 [0.00, 0.31] / 0.092 [0.03, 0.33] / 0.113 [0.06, 0.35] |
| M enc inner val | nms | 0.410 [0.26, 0.58] / 0.497 [0.31, 0.72] / 0.542 [0.38, 0.75] | 0.042 [0.00, 0.22] / 0.092 [0.05, 0.25] / 0.100 [0.07, 0.25] |
| M enc inner val | caps | 0.418 [0.28, 0.57] / 0.521 [0.34, 0.70] / 0.583 [0.43, 0.75] | 0.038 [0.00, 0.21] / 0.075 [0.03, 0.24] / 0.096 [0.08, 0.24] |
| M enc inner val | caps_chain | 0.398 [0.26, 0.57] / 0.501 [0.31, 0.70] / 0.563 [0.41, 0.74] | 0.038 [0.00, 0.21] / 0.075 [0.03, 0.24] / 0.096 [0.08, 0.24] |

| predictions | row acc | row macro F1 | F1@50 P0 | F1@50 P1 |
|---|---|---|---|---|
| M0 enc inner val | 0.463 | 0.256 | 0.2 | 29.0 |
| M enc inner val | 0.433 | 0.237 | 0.2 | 18.0 |

Test:
| predictions | decoder | short mAP@1 / @2 / @5 [95% CI] | long mAP@1 / @2 / @5 [95% CI] |
|---|---|---|---|
| M0 enc test | nms | 0.325 [0.22, 0.45] / 0.395 [0.27, 0.56] / 0.467 [0.35, 0.61] | 0.077 [0.02, 0.16] / 0.113 [0.06, 0.19] / 0.143 [0.11, 0.21] |
| M0 enc test | caps | 0.337 [0.24, 0.46] / 0.408 [0.28, 0.57] / 0.512 [0.39, 0.66] | 0.074 [0.02, 0.15] / 0.117 [0.07, 0.20] / 0.159 [0.11, 0.22] |
| M enc test | nms | 0.321 [0.21, 0.45] / 0.419 [0.30, 0.55] / 0.468 [0.36, 0.59] | 0.111 [0.06, 0.21] / 0.144 [0.11, 0.23] / 0.160 [0.13, 0.25] |
| M enc test | caps | 0.324 [0.21, 0.46] / 0.422 [0.30, 0.56] / 0.494 [0.38, 0.62] | 0.125 [0.06, 0.23] / 0.183 [0.14, 0.27] / 0.216 [0.16, 0.31] |

| predictions | decoder | Grasp | ClipVas | SVart | MetalClip | StartDiss | Denonv | Transect | StitchBl |  (AP@1/@2/@5)
|---|---|---|---|---|---|---|---|---|---|
| M0 enc test | nms | 0.35 0.35 0.49 | 0.28 0.37 0.39 | 0.33 0.42 0.47 | 0.34 0.44 0.52 | 0.02 0.05 0.05 | 0.04 0.12 0.23 | 0.02 0.02 0.03 | 0.23 0.25 0.26 |
| M0 enc test | caps | 0.39 0.39 0.61 | 0.30 0.39 0.45 | 0.33 0.42 0.47 | 0.34 0.44 0.52 | 0.02 0.05 0.05 | 0.03 0.14 0.30 | 0.02 0.02 0.03 | 0.23 0.25 0.25 |
| M enc test | nms | 0.16 0.29 0.36 | 0.36 0.37 0.39 | 0.40 0.49 0.53 | 0.36 0.52 0.58 | 0.05 0.10 0.10 | 0.10 0.18 0.23 | 0.01 0.01 0.02 | 0.28 0.29 0.30 |
| M enc test | caps | 0.18 0.31 0.48 | 0.36 0.37 0.39 | 0.40 0.49 0.53 | 0.36 0.52 0.58 | 0.08 0.17 0.17 | 0.12 0.26 0.37 | 0.01 0.01 0.02 | 0.29 0.30 0.31 |

| predictions | row acc | row macro F1 | F1@50 P0 | F1@50 P1 |
|---|---|---|---|---|
| M0 enc test | 0.490 | 0.255 | 0.1 | 31.4 |
| M enc test | 0.452 | 0.235 | 0.1 | 23.2 |
Decoders chosen on M0 inner val (mAP@2): short caps (0.483 = caps_chain > nms 0.473), long nms (0.096). Selected on test: M0 short caps 0.337 / 0.408 / 0.512, long nms 0.077 / 0.113 / 0.143; M short caps 0.324 / 0.422 / 0.494, long nms 0.111 / 0.144 / 0.160.

## EXP-21 phase 3: fusion, phase decoding, Clip Applier split, online evaluation, confusion matrices, correlations, viewer, exploratory short MS-TCN
Hypothesis: (1) fusing the encoder heads with the long model (per-step confidence weighting, or one global weight)
recovers the long events at least as well as the better of the two alone, for events and phases; (2) the
Clip Applier detected by the teacher makes the clip_ events easier; (3) a causal long model plus a center-clip
encoder with a declared 2 s delay keeps most of the offline long-event mAP online; (4) (exploratory, not
pre-registered) an MS-TCN with the reference EXP-19 S recipe on the M features closes part of the short-event gap to
the reference MS-TCN (0.738 mAP@2) that the per-clip encoder heads leave (0.422).
Parent: 2141846..62c5649 (EXP-21 arms M and M0, long models of phase 2 on arm M features, fold 0).
Base: 90c1af3 (origin/container; the worktree is 3 commits behind its tip 525a397, nothing merged).
Command (run exactly as written, from ~/workspace/wt/focus8-p3, queue of ~/workspace/wt/focus8/run_files/gpuq):
`bash run_files/enqueue_exp21_phase3.sh`, which adds these jobs (cmd of each), seed 0 unless a seed is in the command:
- exp21_p3_mstcn_s0 / _s1 (GPU, priority 196/195): `bash run_files/exp21_p3.sh mstcn <seed> {gpu}` = tools/train_short_mstcn.py
  (reference recipe, last epoch) then tools/eval_focus8.py on inner_val and held_out
- exp21_p3_fuse (CPU, 190): `bash run_files/exp21_p3.sh fuse` = `python tools/fuse_focus8.py --out results/p3/fusion`
- exp21_p3_online (CPU, 189): `bash run_files/exp21_p3.sh online` = `python tools/online_eval_focus8.py --out results/p3/online`
- exp21_p3_clip (188, after fuse): `python tools/clip_applier_split.py --dets results/p3/fusion/final_offline.json --out results/p3/clip_applier/split.json`
- exp21_p3_confusion (187, after fuse and online): `python tools/confusion_focus8.py --out results/p3/confusion --fusion results/p3/fusion --online results/p3/online`
- exp21_p3_corr (186, after fuse): `python tools/corr_focus8.py --out results/p3/corr --fusion results/p3/fusion`
- exp21_p3_viewer (150, after all): `python tools/export_exp21_viewer.py --out /home/gchie/workspace/viewer/EXP21_focus8`
Smoke test (1-2 videos): `bash run_files/exp21_p3_smoke.sh 1`, unit tests: `python tests/test_p3.py`.
Notes: the phase-gating decode arm was already evaluated in phase 2 (no gain) and is not rebuilt. The online
evaluation re-uses the cached center-clip probabilities (identical to streaming: a clip never depends on the
future of its own window beyond the declared 2 s). corr_focus8.py is a new importable module; orsi_corr.py is untouched.
Selection (all on the inner val of fold 0): long model of (b) among a3/sed x seeds and seed ensembles, decoder
(nms / caps / caps_chain), fusion weight, final arm, online decoder and causal model, per-class thresholds
(best F1 at +-2 s) of the confusion matrices; score = long mAP averaged over +-1/2/5 s (phases: F1@50 after P1).
Metric: long-group mAP@+-1/2/5 s of arms (a), (b), (c), (c') on the 10 test videos with bootstrap CI over videos and
the paired per-video Wilcoxon of (c) and (c') against the better of (a) and (b) (chosen on the inner val); phases:
accuracy and F1@50 with and without P1; AP of clip_ events with and without Clip Applier (+-1/2/5 s); online mAP at
+-1/2/5 s, mean latency (emission + 2 s clip delay - ground truth) of the TPs, online confusion matrices;
short-group mAP of the exploratory MS-TCN.
Prediction (written 2026-11-20, before any phase 3 run):
- fusion: best of (c), (c') long mAP@+-2 s >= max((a), (b)) - 0.02 and in 0.30-0.45; Wilcoxon p > 0.05 (10 videos);
- phases: F1@50 with P1 of (c) or (c') >= (b) - 2 points, (b) in 45-60, encoder (a) < 30;
- Clip Applier: mean AP@+-2 s with Clip Applier >= without + 0.10;
- online: long mAP@+-2 s in 0.18-0.32, short mAP@+-2 s 0.30-0.45, mean latency 4.0 +- 0.5 s (2 s decoder + 2 s clip delay);
- exploratory short MS-TCN: short mAP@+-2 s in 0.55-0.75 (above the encoder heads 0.42; reference 0.738).
Stop rule: all jobs run to completion (or fail fast: a failed job blocks only its dependants); no extension to other folds
is decided here: the pre-registered rule of EXP-21 (>= 0.03 mAP@+-2 s over the reference in a group) stays with the analyst.
Machine: Orsi, gpuq (CPU jobs; the MS-TCN on one GPU of 1-3, never GPU0).
Budget: CPU about 1-2 h in total (fusion about 10-20 min, strips and figures a few min), GPU 2 x about 10-20 min.
Data version: labels v2026-11-19 / v2026-11-19_4hz, arm M best.pyth epoch 13, seed 0, long models seeds 0 and 1.
Result (raw numbers; ran 2026-11-20 09:16-09:24 via gpuq, all 8 jobs exit 0; code commit fc6df23, seed 0; outputs in
~/workspace/wt/focus8-p3/results/p3/{fusion,online,clip_applier,confusion,corr,short_mstcn}, viewer
~/workspace/viewer/EXP21_focus8/index.html (curl localhost:8765/EXP21_focus8/ answers)):
- Fusion, long events, test mAP@1/@2/@5: (a) enc nms 0.111/0.144/0.160; (b) a3_s0 nms 0.160/0.357/0.501; (c) a3_s0+enc nms
  0.160/0.316/0.470; (c') w=0.1 nms 0.239/0.410/0.447. Inner-val choices: (b) a3_s0 nms (0.269), (c) 0.211, (c') 0.233,
  (a) 0.078; better of (a),(b) = (b); final long arm = (b). Wilcoxon (per-video long AP) c vs b p = 0.52/0.074/0.57
  (mean diff -0.065/-0.079/-0.022); c' vs b p = 0.36/0.74/0.94 (+0.052/+0.020/-0.023). Short (encoder heads, caps chosen on inner
  val): 0.324/0.422/0.494.
- Phases, test (row acc | F1@50 P0 | F1@50 P1): (a) 0.452 | 0.1 | 23.2; (b) a3_ens 0.778 | 63.1 | 67.8; (c) 0.769 | 24.3 | 67.8;
  (c') w=0.7 0.671 | 1.3 | 58.5. Inner val F1@50 P1: a 18.0, b 53.7, c 52.5, c' 58.6 -> final phase arm = (c'). Wilcoxon F1@50 c vs b:
  P0 p = 0.002 (-38.7), P1 p = 1.0 (0.0); c' vs b: P0 p = 0.002 (-61.8), P1 p = 0.008 (-9.3).
- Clip Applier split (mean of the 3 clip_ classes, AP@1/@2/@5): with 0.344/0.492/0.536, without 0.511/0.554/0.610, all
  0.372/0.459/0.500 (teacher signal >= 0.3 within +-2 s).
- Online (test): short mAP 0.307/0.406/0.440, long 0.123/0.248/0.352 (causal model a3_causal_s0, plain decoder, chosen on the
  inner val); latency of TPs at +-2 s 4.01 s (std 0.85, n 207); online phases F1@50 argmax 33.5 / causal hysteresis 46.0, acc 1 Hz 0.753 / 0.755.
- Exploratory short MS-TCN (reference recipe, last epoch), test mAP@1/@2/@5: seed 0 nms short 0.205/0.567/0.695 caps 0.210/0.593/0.728;
  seed 1 nms 0.319/0.634/0.748 caps 0.322/0.642/0.767 (reference S MS-TCN seed 0: 0.532/0.722/0.848 nms).

## EXP-22: L-A3 on the EXP-19 S encoder features (fold 0), one model for the 8 target events
Hypothesis: the focused fine-tune of EXP-21 (arm M) made the 4 Hz encoder features poorer than the EXP-19 S features, so
the same long-context model (L-A3 = BiMamba2 + 2 MS-TCN stages) trained on the S features beats the reference S MS-TCN
on the long events and phases, and its extra short-event head keeps the reference level on the 4 short events.
Parent: EXP-21 phase 3 (3b955e5, branch exp/focus8-phase3); the S features are the EXP-19 S fold 0 outputs.
Base: origin/main 9eee77e (included in this worktree: 0 commits behind after git fetch; nothing merged).
Data version: labels v2026-11-19 / v2026-11-19_4hz (ORSI_LABEL_DIR as EXP-21); S features
`~/workspace/wt/short-long-context/results/S/LONG/cv_ft/fold<i>/untrimmed/checkpoint_best_steps/all/RARPxx.npz`
(4 Hz, `features` [T, 1536] fp16, same rows and `second` as the arm M features; the S encoder never saw the held-out
videos of its fold). Fold 0 splits (35 train / 5 inner val / 10 held out) = EXP-21 splits (asserted by
tools/s_fold_inputs.py); folds 1-4: train / inner val = `train_data.csv` / `val_data.csv` of the S fine-tune of the
fold (cv_finetune.choose_inner_val), held out = the videos of the EXP-19 S dilate3 prediction dir of the fold.
Arms (2 seeds each, 0 and 1): L-A3 offline and L-A3 causal, tools/train_long_s.py; same architecture, optimiser, schedule
(100 epochs, AdamW 5e-4 cosine, crop 16384 rows, feature dropout 0.5, targets +-1 s, weighted CE, phase CE, 0.15 T-MSE)
as EXP-21 (tools/train_long_mamba.py, imported unchanged). Differences: event head = Idle + 4 short + 4 long targets
(9 outputs); selection on the inner val = mean of the short-group and long-group mAP (each averaged over +-1/2/5 s, NMS).
Reference: EXP-19 S MS-TCN dilate3 seeds 0 and 1, re-evaluated with the EXP-21 evaluator (results/reference/v2026-11-19):
nms, held out: long mAP@+-2 s 0.429 / 0.297 (mean 0.363), short 0.722 / 0.811 (mean 0.767); phases F1@50 P0 / P1
41.3 / 71.3 and 40.3 / 70.4. EXP-21 comparison: arm M L-A3 long 0.357 / 0.303 (nms, @2), F1@50 63 (P0) and 68 (P1).
Command: see "Commands" below (run exactly as written).
Metric: held-out (10 videos) spotting AP per class and group mAP (short, long) at +-1, +-2, +-5 s with per-video bootstrap CI
(tools/eval_focus8.py, 1000 resamples); phase accuracy and F1@50 with (P1) and without (P0) the P1 decoding; per-video paired
Wilcoxon of L-A3 (mean of seeds per video) against the reference MS-TCN; online evaluation of the causal arm with the
EXP-21 online decoding (tools/online_eval_focus8.py functions, clip delay 2 s, decoder delay 2 s).
Prediction (written 2026-11-20, before any run): long-group mAP@+-2 s, NMS decoder, offline L-A3 on S features,
mean of the 2 seeds on the held-out videos >= reference mean + 0.03 = 0.363 + 0.03 = 0.393.
Secondary, not part of the prediction: short-group mAP@+-2 s of the same model within 0.05 of the reference (0.767);
phases F1@50 P1 within 5 points of the reference (70.9 mean).
Stop rule: all runs of fold 0 (4 trainings, 100 epochs each) are done and evaluated, whatever the result. Folds 1-4 are a secondary
cross-validation check (offline L-A3, 2 seeds each, reported after fold 0); no choice is made on them. Every choice (checkpoint
epoch, causal seed / ensemble and online decoder) is made on the inner val only.
Machine: Orsi, gpuq of the focus8 worktree (GPUs 1-3 only), mamba venv for training, Vu-Tapis env for evaluation.
Budget: measured in EXP-21: an L-A3 run takes 7-9 min on one GPU; 4 runs fold 0 plus 8 runs folds 1-4 = about 12 x 8 min = 1.6 GPU h;
evaluation 10-20 CPU min per fold.

Commands (run exactly as written; the queue is `run_files/enqueue_exp22.sh`):
- `bash run_files/exp22_s_long.sh inputs <fold>`   (splits + 9-column reference predictions)
- `bash run_files/exp22_s_long.sh train <fold> <a3|a3_causal> <seed> <gpu>`   (training + eval_focus8 on inner val and held out)
- `bash run_files/exp22_s_long.sh report <fold> [online]`   (tables, Wilcoxon, online)
- `bash run_files/exp22_s_long.sh viewer`   (viewer at ~/workspace/viewer/EXP22_s_long)
Outputs: results/exp22/fold<i>/{a3,a3_causal}_s<seed>/, results/exp22/fold<i>/report/report.{md,json}.

Result:
Experiment commit (section + code, before the runs): a6435ce. Seeds 0 and 1; labels v2026-11-19 / v2026-11-19_4hz; S features of
fold 0 = EXP-19 S fold 0 `untrimmed/checkpoint_best_steps/all` (4 Hz, 1536-d, same `second` as arm M; splits identical to EXP-21).
Runs: results/exp22/fold0/{a3,a3_causal}_s{0,1}/ (train.log, train_log.json, best.pt, inner_val/, held_out/, eval/*.json); each run
7-11 min on one GPU (queue log results/exp22/runs.txt, STATUS in ~/workspace/wt/focus8/results/STATUS.md). Best epoch (inner val):
a3_s0 99 (val metric 0.479), a3_s1 84 (0.517). Report: results/exp22/fold<i>/report/report.{md,json}; tracked copies results_summary/exp22_fold<i>.md.
Viewer: ~/workspace/viewer/EXP22_s_long/index.html (figures/fold0_group_map.png, figures/cv_group_map2.png), linked from EXP21_focus8/index.html.

Fold 0, held-out 10 videos, NMS, group mAP @1 / @2 / @5 (mean of the 2 seeds; per-seed values with 95% CI in the report):
| model | short | long |
|---|---|---|
| S L-A3 offline (EXP-22), seeds 0 / 1 at @2 | 0.566 / 0.708 / 0.771 (0.724, 0.692) | 0.185 / 0.388 / 0.569 (0.419, 0.357) |
| S MS-TCN reference | 0.531 / 0.766 / 0.869 (0.722, 0.811) | 0.215 / 0.363 / 0.601 (0.429, 0.297) |
| M L-A3 offline (EXP-21, long head) | - | 0.129 / 0.330 / 0.500 (0.357, 0.303) |
| M short MS-TCN (EXP-21 p3) | 0.262 / 0.601 / 0.722 (0.567, 0.634) | - |
caps_chain decoder, S L-A3: short 0.565 / 0.706 / 0.770, long 0.192 / 0.393 / 0.590; reference caps_chain: short 0.541 / 0.779 / 0.888, long 0.212 / 0.366 / 0.609.
Prediction check (long mAP@2, NMS, mean of seeds): S L-A3 0.388, reference 0.363, threshold 0.393, difference +0.025.
Per-class AP@1 / @2 / @5, NMS, seed 0 | seed 1 (S L-A3), then reference seed 0 | seed 1:
- Grasping_catheter_tip (n 9): 0.157 0.465 0.561 | 0.074 0.335 0.669 ; ref 0.314 0.330 0.740 | 0.391 0.781 0.944
- Clip_or_coagulate_vas_deferens (17): 0.715 0.774 0.777 | 0.694 0.778 0.779 ; ref 0.487 0.844 0.845 | 0.528 0.706 0.783
- Identification_and_clipping_SV_arteries (50): 0.738 0.813 0.813 | 0.653 0.811 0.847 ; ref 0.621 0.806 0.872 | 0.512 0.848 0.894
- Metal_clip_on_right_pedicle (123): 0.745 0.842 0.858 | 0.751 0.843 0.860 ; ref 0.707 0.907 0.936 | 0.687 0.908 0.938
- Start_dissection (10): 0.166 0.288 0.600 | 0.351 0.399 0.503 ; ref 0.317 0.454 0.768 | 0.137 0.323 0.702
- Incision_of_Denonvilliers_fascia (9): 0.303 0.478 0.639 | 0.040 0.370 0.636 ; ref 0.367 0.618 0.703 | 0.305 0.455 0.753
- Transection_of_the_urethra (10): 0.049 0.301 0.508 | 0.241 0.435 0.445 ; ref 0.091 0.252 0.531 | 0.242 0.309 0.506
- Stitch_in_bladder (10): 0.252 0.608 0.681 | 0.080 0.226 0.543 ; ref 0.193 0.391 0.441 | 0.069 0.100 0.402
Phases, held-out (row acc / row macro F1 / F1@50 P0 / F1@50 P1): S L-A3 s0 0.863 / 0.747 / 75.5 / 79.6, s1 0.855 / 0.740 / 69.0 / 77.6;
reference s0 0.835 / 0.713 / 41.3 / 71.3, s1 0.832 / 0.710 / 40.3 / 70.4; M L-A3 s0 0.768 / 0.618 / 51.9 / 71.0, s1 0.788 / 0.665 / 58.8 / 69.2.
Paired Wilcoxon, 10 videos, L-A3 S (seed mean per video) minus reference (seed mean), NMS (mean diff, better/worse, p):
short @1 +0.008 7/3 p 0.770; short @2 -0.070 2/8 p 0.020; short @5 -0.113 0/10 p 0.002; long @1 +0.001 5/5 p 0.748; long @2 +0.030 6/3 p 0.359;
long @5 -0.014 4/6 p 0.625; phases F1@50 P0 +31.5 10/0 p 0.002; P1 +7.7 8/1 p 0.008.
Online, causal L-A3 S (clip delay 2 s, decoder delay 2 s; inner-val selection: a3_causal_s0, plain decoder; phases: seed ensemble a3_causal_ens):
short 0.275 / 0.494 / 0.742 [@2 CI 0.43, 0.63], long 0.292 / 0.364 / 0.587 [@2 CI 0.26, 0.54]; per seed plain: s0 short 0.275 / 0.494 / 0.742, long 0.292 / 0.364 / 0.587;
s1 short 0.238 / 0.526 / 0.680, long 0.169 / 0.286 / 0.424; caps_chain online s0 short 0.254 / 0.415 / 0.566, long 0.139 / 0.188 / 0.325, s1 short 0.203 / 0.443 / 0.564, long 0.092 / 0.173 / 0.280.
Latency of the true positives (s): @1 mean 4.13 std 0.55 n 158; @2 mean 4.11 std 0.81 n 195; @5 mean 3.88 std 1.62 n 229. Phases online F1@50 P0 44.7, H 59.2.
EXP-21 online (same evaluation): short 0.307 / 0.406 / 0.440, long 0.123 / 0.248 / 0.352.

Folds 1-4 (secondary CV check, offline L-A3, 2 seeds, same settings, held-out 10 videos each; NMS, mean of seeds, group mAP @1 / @2 / @5):
| fold | S L-A3 short | reference short | S L-A3 long | reference long | long @2 diff (threshold +0.03) | F1@50 P0/P1 S L-A3 s0, s1 | reference s0, s1 |
|---|---|---|---|---|---|---|---|
| 1 | 0.577 / 0.719 / 0.820 | 0.500 / 0.772 / 0.879 | 0.250 / 0.406 / 0.539 | 0.247 / 0.394 / 0.537 | +0.012 | 72.0/77.4, 71.0/78.9 | 42.2/74.1, 40.5/74.9 |
| 2 | 0.570 / 0.753 / 0.855 | 0.521 / 0.770 / 0.855 | 0.274 / 0.442 / 0.535 | 0.234 / 0.354 / 0.531 | +0.088 | 54.8/66.4, 74.2/76.9 | 44.0/73.4, 47.1/72.4 |
| 3 | 0.566 / 0.745 / 0.843 | 0.504 / 0.833 / 0.892 | 0.141 / 0.425 / 0.557 | 0.184 / 0.297 / 0.536 | +0.128 | 54.5/69.5, 72.0/80.4 | 45.1/75.5, 36.2/73.1 |
| 4 | 0.548 / 0.768 / 0.869 | 0.483 / 0.763 / 0.917 | 0.140 / 0.354 / 0.574 | 0.100 / 0.303 / 0.528 | +0.052 | 70.2/77.6, 72.7/83.0 | 45.6/73.5, 49.5/75.0 |
Wilcoxon per fold (long @2 mean diff, p; short @2 mean diff, p; F1@50 P1 mean diff, p): fold 1 +0.069 p 0.049; -0.032 p 0.375; +3.7 p 0.055.
fold 2 +0.035 p 0.846; -0.027 p 0.846; -1.2 p 0.432. fold 3 +0.089 p 0.160; -0.065 p 0.014; +0.6 p 0.910. fold 4 +0.074 p 0.432; +0.052 p 0.322; +6.0 p 0.055.
Long @2 differences are the L-A3 mean minus the reference mean (prediction rule applied per fold, not pre-registered): fold 0 +0.025, 1 +0.012, 2 +0.088, 3 +0.128, 4 +0.052 (mean over the 5 folds +0.061).
Notes: no GPU0 use (queue GPUs 1-3); GPU use 86-97% during training; evaluation CPU only. Selection score on the inner val = mean of short and long group mAP (see Arms), not the long group alone as in EXP-21.
