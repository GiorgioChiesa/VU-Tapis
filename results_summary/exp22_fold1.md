# EXP-22 fold 1: L-A3 on EXP-19 S latents vs reference (held-out, 10 videos, 95% CI = per-video bootstrap)

## Group mAP, decoder nms (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.556 [0.46, 0.70] / 0.695 [0.59, 0.83] / 0.837 [0.77, 0.90] | 0.262 [0.14, 0.46] / 0.453 [0.34, 0.64] / 0.573 [0.49, 0.70] |
| S L-A3 offline | seed 1 | 0.598 [0.49, 0.75] / 0.744 [0.65, 0.87] / 0.802 [0.71, 0.92] | 0.238 [0.13, 0.40] / 0.359 [0.29, 0.49] / 0.504 [0.42, 0.64] |
| S L-A3 offline | mean of seeds | 0.577 / 0.719 / 0.820 | 0.250 / 0.406 / 0.539 |
| S MS-TCN reference | seed 0 | 0.515 [0.42, 0.63] / 0.786 [0.68, 0.89] / 0.883 [0.82, 0.94] | 0.258 [0.20, 0.38] / 0.385 [0.28, 0.52] / 0.520 [0.44, 0.64] |
| S MS-TCN reference | seed 1 | 0.486 [0.39, 0.62] / 0.758 [0.66, 0.88] / 0.875 [0.79, 0.95] | 0.235 [0.15, 0.36] / 0.403 [0.34, 0.49] / 0.554 [0.48, 0.67] |
| S MS-TCN reference | mean of seeds | 0.500 / 0.772 / 0.879 | 0.247 / 0.394 / 0.537 |

## Group mAP, decoder caps_chain (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.553 [0.45, 0.70] / 0.694 [0.59, 0.83] / 0.835 [0.77, 0.91] | 0.259 [0.14, 0.47] / 0.448 [0.33, 0.64] / 0.562 [0.47, 0.70] |
| S L-A3 offline | seed 1 | 0.593 [0.48, 0.75] / 0.743 [0.65, 0.87] / 0.801 [0.71, 0.92] | 0.240 [0.13, 0.40] / 0.360 [0.29, 0.50] / 0.505 [0.40, 0.66] |
| S L-A3 offline | mean of seeds | 0.573 / 0.718 / 0.818 | 0.250 / 0.404 / 0.533 |
| S MS-TCN reference | seed 0 | 0.516 [0.42, 0.63] / 0.792 [0.69, 0.90] / 0.891 [0.83, 0.95] | 0.256 [0.20, 0.37] / 0.383 [0.28, 0.52] / 0.542 [0.44, 0.69] |
| S MS-TCN reference | seed 1 | 0.492 [0.39, 0.62] / 0.765 [0.67, 0.88] / 0.878 [0.79, 0.96] | 0.233 [0.15, 0.35] / 0.400 [0.34, 0.49] / 0.566 [0.49, 0.69] |
| S MS-TCN reference | mean of seeds | 0.504 / 0.778 / 0.885 | 0.244 / 0.392 / 0.554 |

## AP per class, decoder nms (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 10 | 0.284 [0.05, 0.68] / 0.437 [0.13, 0.81] / 0.759 [0.50, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.579 [0.35, 0.83] / 0.717 [0.52, 0.93] / 0.863 [0.73, 0.96] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.706 [0.62, 0.81] / 0.831 [0.79, 0.88] / 0.866 [0.82, 0.91] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.657 [0.53, 0.77] / 0.792 [0.69, 0.88] / 0.862 [0.81, 0.91] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.237 [0.03, 0.62] / 0.367 [0.16, 0.72] / 0.377 [0.18, 0.72] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.445 [0.15, 0.88] / 0.649 [0.36, 1.00] / 0.735 [0.41, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 9 | 0.156 [0.00, 0.48] / 0.544 [0.20, 0.88] / 0.874 [0.67, 1.00] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 9 | 0.208 [0.01, 0.53] / 0.253 [0.04, 0.56] / 0.305 [0.09, 0.60] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 10 | 0.472 [0.24, 0.81] / 0.607 [0.36, 0.91] / 0.608 [0.36, 0.91] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.637 [0.41, 0.93] / 0.714 [0.52, 0.94] / 0.832 [0.69, 0.97] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.716 [0.62, 0.82] / 0.871 [0.83, 0.92] / 0.895 [0.86, 0.94] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.568 [0.44, 0.68] / 0.786 [0.66, 0.88] / 0.873 [0.83, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.135 [0.00, 0.37] / 0.274 [0.08, 0.58] / 0.431 [0.21, 0.69] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.568 [0.22, 1.00] / 0.660 [0.26, 1.00] / 0.816 [0.49, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 9 | 0.148 [0.00, 0.44] / 0.328 [0.09, 0.71] / 0.588 [0.38, 0.89] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 9 | 0.099 [0.01, 0.39] / 0.175 [0.05, 0.47] / 0.183 [0.07, 0.47] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 10 | 0.509 [0.18, 0.86] / 0.777 [0.51, 1.00] / 0.887 [0.71, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.466 [0.27, 0.70] / 0.730 [0.58, 0.89] / 0.840 [0.77, 0.94] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.579 [0.49, 0.70] / 0.863 [0.78, 0.93] / 0.898 [0.82, 0.96] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.507 [0.37, 0.64] / 0.774 [0.64, 0.88] / 0.906 [0.85, 0.95] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.220 [0.03, 0.57] / 0.420 [0.13, 0.78] / 0.556 [0.35, 0.84] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.380 [0.06, 0.75] / 0.555 [0.23, 0.89] / 0.813 [0.57, 0.99] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 9 | 0.395 [0.11, 0.70] / 0.503 [0.20, 0.80] / 0.600 [0.38, 0.85] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 9 | 0.039 [0.00, 0.25] / 0.062 [0.01, 0.28] / 0.113 [0.04, 0.33] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 10 | 0.409 [0.13, 0.81] / 0.544 [0.23, 0.91] / 0.840 [0.65, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.582 [0.35, 0.83] / 0.812 [0.67, 0.96] / 0.855 [0.76, 0.96] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.461 [0.39, 0.55] / 0.885 [0.80, 0.95] / 0.900 [0.83, 0.96] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.492 [0.37, 0.60] / 0.790 [0.70, 0.87] / 0.903 [0.85, 0.95] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.260 [0.00, 0.59] / 0.334 [0.07, 0.66] / 0.516 [0.27, 0.83] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.271 [0.05, 0.68] / 0.739 [0.47, 0.95] / 0.822 [0.57, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 9 | 0.394 [0.12, 0.70] / 0.480 [0.19, 0.77] / 0.762 [0.51, 0.96] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 9 | 0.015 [0.00, 0.09] / 0.060 [0.00, 0.23] / 0.116 [0.04, 0.29] |

## AP per class, decoder caps_chain (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 10 | 0.279 [0.05, 0.68] / 0.431 [0.13, 0.81] / 0.746 [0.45, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.571 [0.33, 0.83] / 0.720 [0.51, 0.93] / 0.866 [0.71, 0.98] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.706 [0.62, 0.81] / 0.831 [0.79, 0.88] / 0.866 [0.82, 0.91] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.657 [0.53, 0.77] / 0.792 [0.69, 0.88] / 0.862 [0.81, 0.91] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.232 [0.03, 0.62] / 0.351 [0.14, 0.73] / 0.361 [0.16, 0.73] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.425 [0.13, 0.87] / 0.635 [0.33, 1.00] / 0.712 [0.38, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 9 | 0.155 [0.00, 0.48] / 0.544 [0.20, 0.88] / 0.873 [0.66, 1.00] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 9 | 0.223 [0.01, 0.56] / 0.262 [0.05, 0.58] / 0.301 [0.08, 0.62] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 10 | 0.457 [0.21, 0.81] / 0.594 [0.30, 0.91] / 0.595 [0.31, 0.91] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.632 [0.41, 0.92] / 0.721 [0.52, 0.96] / 0.841 [0.70, 0.98] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.716 [0.62, 0.82] / 0.871 [0.83, 0.92] / 0.895 [0.86, 0.94] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.568 [0.44, 0.68] / 0.786 [0.66, 0.88] / 0.873 [0.83, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.135 [0.00, 0.37] / 0.253 [0.07, 0.56] / 0.359 [0.17, 0.64] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.568 [0.22, 1.00] / 0.660 [0.26, 1.00] / 0.816 [0.49, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 9 | 0.156 [0.00, 0.46] / 0.357 [0.10, 0.76] / 0.668 [0.42, 1.00] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 9 | 0.103 [0.01, 0.40] / 0.170 [0.05, 0.47] / 0.177 [0.06, 0.47] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 10 | 0.519 [0.18, 0.88] / 0.790 [0.51, 1.00] / 0.912 [0.73, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.460 [0.27, 0.70] / 0.739 [0.58, 0.91] / 0.850 [0.77, 0.96] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.579 [0.49, 0.70] / 0.863 [0.78, 0.93] / 0.898 [0.82, 0.96] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.507 [0.37, 0.64] / 0.774 [0.64, 0.88] / 0.906 [0.85, 0.95] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.220 [0.03, 0.57] / 0.429 [0.13, 0.80] / 0.625 [0.39, 1.00] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.380 [0.06, 0.75] / 0.554 [0.23, 0.89] / 0.797 [0.55, 0.99] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 9 | 0.384 [0.11, 0.69] / 0.488 [0.18, 0.80] / 0.621 [0.37, 0.89] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 9 | 0.039 [0.00, 0.25] / 0.062 [0.01, 0.28] / 0.123 [0.04, 0.38] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 10 | 0.433 [0.14, 0.84] / 0.572 [0.24, 1.00] / 0.854 [0.62, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 21 | 0.580 [0.35, 0.83] / 0.813 [0.67, 0.96] / 0.857 [0.76, 0.96] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 95 | 0.461 [0.39, 0.55] / 0.885 [0.80, 0.95] / 0.900 [0.83, 0.96] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.492 [0.37, 0.60] / 0.790 [0.70, 0.87] / 0.903 [0.85, 0.95] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.260 [0.00, 0.59] / 0.334 [0.07, 0.66] / 0.541 [0.28, 0.88] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.276 [0.05, 0.70] / 0.741 [0.45, 1.00] / 0.827 [0.56, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 9 | 0.385 [0.11, 0.69] / 0.481 [0.18, 0.78] / 0.774 [0.50, 0.97] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 9 | 0.010 [0.00, 0.06] / 0.044 [0.00, 0.22] / 0.123 [0.04, 0.36] |

## Phases (held-out)
| model | run | row acc | row macro F1 | F1@50 P0 | F1@50 P1 | acc 1 Hz P0 | acc 1 Hz P1 |
|---|---|---|---|---|---|---|---|
| S L-A3 offline | seed 0 | 0.871 | 0.752 | 72.0 | 77.4 | 0.871 | 0.873 |
| S L-A3 offline | seed 1 | 0.846 | 0.726 | 71.0 | 78.9 | 0.847 | 0.847 |
| S MS-TCN reference | seed 0 | 0.838 | 0.723 | 42.2 | 74.1 | 0.838 | 0.836 |
| S MS-TCN reference | seed 1 | 0.835 | 0.720 | 40.5 | 74.9 | 0.835 | 0.837 |

## Paired Wilcoxon over the 10 held-out videos (L-A3 S, mean of seeds per video, minus the other model)
| key | n | mean diff | median diff | better | worse | p |
|---|---|---|---|---|---|---|
| nms|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.080 | +0.099 | 9 | 1 | 0.064 |
| nms|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.032 | -0.027 | 4 | 6 | 0.375 |
| nms|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.048 | -0.037 | 0 | 10 | 0.002 |
| nms|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.037 | -0.027 | 3 | 7 | 0.625 |
| nms|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.069 | +0.049 | 8 | 2 | 0.049 |
| nms|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.021 | -0.003 | 3 | 6 | 0.359 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.080 | +0.099 | 9 | 1 | 0.064 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.032 | -0.027 | 4 | 6 | 0.375 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.048 | -0.037 | 0 | 10 | 0.002 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.037 | -0.027 | 3 | 7 | 0.625 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.069 | +0.049 | 8 | 2 | 0.049 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.021 | -0.003 | 3 | 6 | 0.359 |
| phases F1@50 P0|S L-A3 offline vs S MS-TCN reference | 10 | +30.162 | +36.332 | 10 | 0 | 0.002 |
| phases F1@50 P1|S L-A3 offline vs S MS-TCN reference | 10 | +3.651 | +5.962 | 6 | 3 | 0.055 |

## Prediction check: long-group mAP@+-2 s, mean of 2 seeds, S L-A3 offline vs reference + 0.03
| decoder | S L-A3 seeds | S L-A3 mean | reference seeds | reference mean | threshold | difference |
|---|---|---|---|---|---|---|
| nms | 0.453, 0.359 | 0.406 | 0.385, 0.403 | 0.394 | 0.424 | +0.012 |
| caps_chain | 0.448, 0.360 | 0.404 | 0.383, 0.400 | 0.392 | 0.422 | +0.012 |
