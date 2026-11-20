# EXP-22 fold 4: L-A3 on EXP-19 S latents vs reference (held-out, 10 videos, 95% CI = per-video bootstrap)

## Group mAP, decoder nms (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.519 [0.42, 0.67] / 0.753 [0.64, 0.88] / 0.857 [0.78, 0.94] | 0.115 [0.06, 0.32] / 0.271 [0.16, 0.49] / 0.583 [0.52, 0.71] |
| S L-A3 offline | seed 1 | 0.576 [0.46, 0.75] / 0.784 [0.68, 0.92] / 0.881 [0.83, 0.94] | 0.165 [0.06, 0.39] / 0.438 [0.34, 0.61] / 0.566 [0.50, 0.67] |
| S L-A3 offline | mean of seeds | 0.548 / 0.768 / 0.869 | 0.140 / 0.354 / 0.574 |
| S MS-TCN reference | seed 0 | 0.496 [0.39, 0.64] / 0.750 [0.66, 0.85] / 0.917 [0.87, 0.97] | 0.089 [0.04, 0.24] / 0.331 [0.22, 0.54] / 0.544 [0.49, 0.65] |
| S MS-TCN reference | seed 1 | 0.470 [0.38, 0.60] / 0.775 [0.66, 0.90] / 0.918 [0.86, 0.98] | 0.112 [0.05, 0.28] / 0.274 [0.21, 0.40] / 0.513 [0.46, 0.61] |
| S MS-TCN reference | mean of seeds | 0.483 / 0.763 / 0.917 | 0.100 / 0.303 / 0.528 |

## Group mAP, decoder caps_chain (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.521 [0.42, 0.67] / 0.763 [0.66, 0.89] / 0.866 [0.80, 0.94] | 0.121 [0.06, 0.31] / 0.282 [0.16, 0.51] / 0.594 [0.53, 0.72] |
| S L-A3 offline | seed 1 | 0.577 [0.46, 0.75] / 0.792 [0.70, 0.92] / 0.874 [0.82, 0.94] | 0.169 [0.07, 0.39] / 0.429 [0.34, 0.59] / 0.567 [0.50, 0.68] |
| S L-A3 offline | mean of seeds | 0.549 / 0.778 / 0.870 | 0.145 / 0.356 / 0.580 |
| S MS-TCN reference | seed 0 | 0.496 [0.39, 0.64] / 0.754 [0.66, 0.85] / 0.923 [0.88, 0.97] | 0.077 [0.03, 0.23] / 0.315 [0.20, 0.53] / 0.549 [0.49, 0.67] |
| S MS-TCN reference | seed 1 | 0.469 [0.38, 0.60] / 0.769 [0.66, 0.89] / 0.915 [0.85, 0.98] | 0.106 [0.04, 0.27] / 0.268 [0.21, 0.40] / 0.521 [0.46, 0.62] |
| S MS-TCN reference | mean of seeds | 0.482 / 0.762 / 0.919 | 0.092 / 0.291 / 0.535 |

## AP per class, decoder nms (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 9 | 0.102 [0.01, 0.42] / 0.634 [0.27, 1.00] / 0.931 [0.78, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.674 [0.37, 1.00] / 0.822 [0.62, 1.00] / 0.822 [0.62, 1.00] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.664 [0.57, 0.78] / 0.770 [0.65, 0.90] / 0.813 [0.69, 0.93] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 172 | 0.637 [0.53, 0.74] / 0.786 [0.71, 0.87] / 0.865 [0.78, 0.93] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.178 [0.01, 0.54] / 0.236 [0.05, 0.61] / 0.902 [0.70, 1.00] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 10 | 0.158 [0.01, 0.53] / 0.528 [0.20, 0.93] / 0.834 [0.63, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.035 [0.00, 0.18] / 0.074 [0.02, 0.32] / 0.252 [0.10, 0.62] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 10 | 0.089 [0.01, 0.28] / 0.246 [0.11, 0.51] / 0.342 [0.23, 0.54] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 9 | 0.285 [0.03, 0.67] / 0.618 [0.31, 1.00] / 0.933 [0.79, 1.00] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.651 [0.42, 0.89] / 0.908 [0.79, 1.00] / 0.908 [0.79, 1.00] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.676 [0.57, 0.80] / 0.771 [0.66, 0.89] / 0.809 [0.69, 0.93] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 172 | 0.692 [0.61, 0.78] / 0.838 [0.77, 0.90] / 0.871 [0.82, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.297 [0.06, 0.75] / 0.558 [0.24, 0.91] / 0.801 [0.53, 1.00] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 10 | 0.121 [0.01, 0.46] / 0.691 [0.39, 1.00] / 0.920 [0.77, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.188 [0.06, 0.54] / 0.371 [0.13, 0.69] / 0.404 [0.16, 0.73] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 10 | 0.056 [0.00, 0.30] / 0.130 [0.02, 0.45] / 0.138 [0.04, 0.45] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 9 | 0.056 [0.00, 0.30] / 0.491 [0.16, 0.83] / 0.972 [0.92, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.701 [0.32, 1.00] / 0.835 [0.62, 1.00] / 0.868 [0.69, 1.00] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.548 [0.46, 0.67] / 0.817 [0.72, 0.91] / 0.893 [0.79, 0.97] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 172 | 0.677 [0.59, 0.77] / 0.858 [0.81, 0.91] / 0.933 [0.90, 0.96] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.138 [0.01, 0.52] / 0.315 [0.07, 0.76] / 0.861 [0.67, 1.00] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 10 | 0.040 [0.00, 0.23] / 0.727 [0.49, 0.95] / 0.824 [0.58, 1.00] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.148 [0.03, 0.43] / 0.167 [0.05, 0.45] / 0.356 [0.26, 0.59] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 10 | 0.030 [0.01, 0.08] / 0.115 [0.05, 0.38] / 0.133 [0.07, 0.39] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 9 | 0.236 [0.00, 0.53] / 0.597 [0.27, 0.94] / 0.929 [0.78, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.593 [0.31, 0.94] / 0.838 [0.62, 1.00] / 0.910 [0.79, 1.00] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.468 [0.32, 0.64] / 0.791 [0.66, 0.90] / 0.881 [0.74, 0.97] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 172 | 0.582 [0.55, 0.64] / 0.875 [0.81, 0.93] / 0.950 [0.92, 0.98] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.130 [0.00, 0.45] / 0.261 [0.04, 0.66] / 0.791 [0.53, 1.00] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 10 | 0.176 [0.01, 0.57] / 0.615 [0.28, 0.94] / 0.858 [0.67, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.107 [0.02, 0.33] / 0.126 [0.05, 0.34] / 0.301 [0.17, 0.53] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 10 | 0.035 [0.01, 0.11] / 0.095 [0.06, 0.27] / 0.101 [0.06, 0.27] |

## AP per class, decoder caps_chain (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 9 | 0.111 [0.01, 0.44] / 0.678 [0.32, 1.00] / 0.967 [0.86, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.673 [0.36, 1.00] / 0.819 [0.62, 1.00] / 0.819 [0.62, 1.00] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.664 [0.57, 0.78] / 0.770 [0.65, 0.90] / 0.813 [0.69, 0.93] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 172 | 0.637 [0.53, 0.74] / 0.786 [0.71, 0.87] / 0.865 [0.78, 0.93] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.178 [0.01, 0.54] / 0.233 [0.05, 0.61] / 0.902 [0.70, 1.00] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 10 | 0.183 [0.01, 0.57] / 0.553 [0.20, 1.00] / 0.845 [0.63, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.037 [0.00, 0.18] / 0.091 [0.02, 0.37] / 0.298 [0.10, 0.68] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 10 | 0.084 [0.01, 0.29] / 0.249 [0.11, 0.55] / 0.330 [0.21, 0.60] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 9 | 0.290 [0.04, 0.69] / 0.663 [0.34, 1.00] / 0.918 [0.74, 1.00] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.651 [0.42, 0.89] / 0.898 [0.77, 1.00] / 0.898 [0.77, 1.00] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.676 [0.57, 0.80] / 0.771 [0.66, 0.89] / 0.809 [0.69, 0.93] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 172 | 0.692 [0.61, 0.78] / 0.838 [0.77, 0.90] / 0.871 [0.82, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.316 [0.06, 0.77] / 0.539 [0.21, 0.91] / 0.812 [0.53, 1.00] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 10 | 0.121 [0.01, 0.46] / 0.691 [0.39, 1.00] / 0.920 [0.77, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.182 [0.06, 0.52] / 0.361 [0.12, 0.69] / 0.400 [0.15, 0.72] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 10 | 0.056 [0.00, 0.30] / 0.126 [0.02, 0.45] / 0.135 [0.04, 0.45] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 9 | 0.056 [0.00, 0.30] / 0.506 [0.18, 0.86] / 1.000 [1.00, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.701 [0.32, 1.00] / 0.835 [0.62, 1.00] / 0.868 [0.69, 1.00] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.548 [0.46, 0.67] / 0.817 [0.72, 0.91] / 0.893 [0.79, 0.97] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 172 | 0.677 [0.59, 0.77] / 0.858 [0.81, 0.91] / 0.933 [0.90, 0.96] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.138 [0.01, 0.52] / 0.315 [0.07, 0.76] / 0.908 [0.73, 1.00] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 10 | 0.033 [0.00, 0.19] / 0.707 [0.45, 0.95] / 0.824 [0.58, 1.00] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.114 [0.02, 0.40] / 0.133 [0.04, 0.42] / 0.338 [0.24, 0.59] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 10 | 0.025 [0.01, 0.07] / 0.105 [0.04, 0.37] / 0.123 [0.06, 0.39] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 9 | 0.235 [0.00, 0.52] / 0.576 [0.24, 0.92] / 0.926 [0.77, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 14 | 0.593 [0.31, 0.94] / 0.833 [0.61, 1.00] / 0.904 [0.78, 1.00] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 88 | 0.468 [0.32, 0.64] / 0.791 [0.66, 0.90] / 0.881 [0.74, 0.97] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 172 | 0.582 [0.55, 0.64] / 0.875 [0.81, 0.93] / 0.950 [0.92, 0.98] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.130 [0.00, 0.45] / 0.259 [0.04, 0.65] / 0.837 [0.58, 1.00] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 10 | 0.183 [0.01, 0.59] / 0.623 [0.29, 1.00] / 0.861 [0.66, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.078 [0.02, 0.28] / 0.097 [0.04, 0.29] / 0.289 [0.16, 0.53] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 10 | 0.033 [0.01, 0.11] / 0.093 [0.06, 0.27] / 0.098 [0.06, 0.27] |

## Phases (held-out)
| model | run | row acc | row macro F1 | F1@50 P0 | F1@50 P1 | acc 1 Hz P0 | acc 1 Hz P1 |
|---|---|---|---|---|---|---|---|
| S L-A3 offline | seed 0 | 0.851 | 0.734 | 70.2 | 77.6 | 0.864 | 0.863 |
| S L-A3 offline | seed 1 | 0.873 | 0.756 | 72.7 | 83.0 | 0.880 | 0.880 |
| S MS-TCN reference | seed 0 | 0.848 | 0.727 | 45.6 | 73.5 | 0.860 | 0.861 |
| S MS-TCN reference | seed 1 | 0.857 | 0.735 | 49.5 | 75.0 | 0.869 | 0.869 |

## Paired Wilcoxon over the 10 held-out videos (L-A3 S, mean of seeds per video, minus the other model)
| key | n | mean diff | median diff | better | worse | p |
|---|---|---|---|---|---|---|
| nms|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.107 | +0.135 | 8 | 2 | 0.084 |
| nms|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | +0.052 | +0.033 | 5 | 5 | 0.322 |
| nms|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.034 | -0.016 | 1 | 9 | 0.014 |
| nms|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.080 | +0.108 | 6 | 4 | 0.492 |
| nms|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.074 | +0.112 | 7 | 3 | 0.432 |
| nms|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | +0.039 | +0.051 | 7 | 3 | 0.322 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.107 | +0.135 | 8 | 2 | 0.084 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | +0.052 | +0.033 | 5 | 5 | 0.322 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.034 | -0.016 | 1 | 9 | 0.014 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.080 | +0.108 | 6 | 4 | 0.492 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.074 | +0.112 | 7 | 3 | 0.432 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | +0.039 | +0.051 | 7 | 3 | 0.322 |
| phases F1@50 P0|S L-A3 offline vs S MS-TCN reference | 10 | +23.905 | +22.808 | 9 | 1 | 0.004 |
| phases F1@50 P1|S L-A3 offline vs S MS-TCN reference | 10 | +6.018 | +6.568 | 7 | 2 | 0.055 |

## Prediction check: long-group mAP@+-2 s, mean of 2 seeds, S L-A3 offline vs reference + 0.03
| decoder | S L-A3 seeds | S L-A3 mean | reference seeds | reference mean | threshold | difference |
|---|---|---|---|---|---|---|
| nms | 0.271, 0.438 | 0.354 | 0.331, 0.274 | 0.303 | 0.333 | +0.052 |
| caps_chain | 0.282, 0.429 | 0.356 | 0.315, 0.268 | 0.291 | 0.321 | +0.064 |
