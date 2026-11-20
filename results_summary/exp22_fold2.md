# EXP-22 fold 2: L-A3 on EXP-19 S latents vs reference (held-out, 10 videos, 95% CI = per-video bootstrap)

## Group mAP, decoder nms (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.516 [0.40, 0.68] / 0.726 [0.62, 0.86] / 0.845 [0.77, 0.92] | 0.216 [0.13, 0.40] / 0.369 [0.24, 0.56] / 0.457 [0.32, 0.65] |
| S L-A3 offline | seed 1 | 0.623 [0.50, 0.77] / 0.780 [0.66, 0.91] / 0.865 [0.78, 0.94] | 0.333 [0.16, 0.60] / 0.515 [0.33, 0.75] / 0.612 [0.43, 0.82] |
| S L-A3 offline | mean of seeds | 0.570 / 0.753 / 0.855 | 0.274 / 0.442 / 0.535 |
| S MS-TCN reference | seed 0 | 0.585 [0.46, 0.74] / 0.776 [0.67, 0.90] / 0.853 [0.77, 0.94] | 0.262 [0.15, 0.45] / 0.314 [0.21, 0.52] / 0.549 [0.40, 0.74] |
| S MS-TCN reference | seed 1 | 0.457 [0.36, 0.61] / 0.764 [0.64, 0.91] / 0.857 [0.76, 0.94] | 0.207 [0.09, 0.40] / 0.395 [0.26, 0.58] / 0.512 [0.40, 0.68] |
| S MS-TCN reference | mean of seeds | 0.521 / 0.770 / 0.855 | 0.234 / 0.354 / 0.531 |

## Group mAP, decoder caps_chain (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.516 [0.40, 0.68] / 0.726 [0.61, 0.86] / 0.852 [0.78, 0.93] | 0.206 [0.12, 0.39] / 0.356 [0.23, 0.55] / 0.440 [0.29, 0.65] |
| S L-A3 offline | seed 1 | 0.618 [0.49, 0.77] / 0.775 [0.65, 0.91] / 0.860 [0.77, 0.94] | 0.343 [0.17, 0.61] / 0.519 [0.33, 0.76] / 0.625 [0.44, 0.83] |
| S L-A3 offline | mean of seeds | 0.567 / 0.751 / 0.856 | 0.275 / 0.438 / 0.532 |
| S MS-TCN reference | seed 0 | 0.569 [0.44, 0.73] / 0.758 [0.63, 0.90] / 0.850 [0.76, 0.94] | 0.261 [0.15, 0.45] / 0.328 [0.21, 0.55] / 0.604 [0.45, 0.78] |
| S MS-TCN reference | seed 1 | 0.459 [0.36, 0.60] / 0.747 [0.61, 0.90] / 0.849 [0.75, 0.94] | 0.208 [0.08, 0.41] / 0.406 [0.24, 0.61] / 0.537 [0.40, 0.73] |
| S MS-TCN reference | mean of seeds | 0.514 / 0.752 / 0.849 | 0.234 / 0.367 / 0.570 |

## AP per class, decoder nms (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 9 | 0.253 [0.04, 0.67] / 0.618 [0.29, 0.96] / 0.830 [0.56, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.669 [0.44, 0.93] / 0.720 [0.53, 0.93] / 0.849 [0.71, 0.96] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.587 [0.43, 0.74] / 0.736 [0.59, 0.86] / 0.823 [0.71, 0.93] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.555 [0.42, 0.68] / 0.828 [0.75, 0.90] / 0.879 [0.81, 0.93] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.147 [0.03, 0.50] / 0.609 [0.31, 0.88] / 0.725 [0.48, 0.92] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.232 [0.03, 0.57] / 0.327 [0.10, 0.65] / 0.390 [0.18, 0.71] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.111 [0.03, 0.28] / 0.112 [0.03, 0.28] / 0.283 [0.13, 0.62] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 9 | 0.375 [0.16, 0.67] / 0.427 [0.22, 0.72] / 0.430 [0.23, 0.72] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 9 | 0.362 [0.07, 0.73] / 0.478 [0.16, 0.86] / 0.767 [0.46, 1.00] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.831 [0.62, 0.98] / 0.922 [0.83, 0.98] / 0.922 [0.83, 0.98] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.688 [0.59, 0.82] / 0.872 [0.78, 0.97] / 0.893 [0.81, 0.97] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.612 [0.50, 0.72] / 0.848 [0.78, 0.90] / 0.879 [0.83, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.305 [0.07, 0.74] / 0.754 [0.46, 0.98] / 0.896 [0.73, 0.99] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.350 [0.06, 0.71] / 0.460 [0.16, 0.82] / 0.552 [0.27, 0.89] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.132 [0.02, 0.44] / 0.303 [0.09, 0.65] / 0.450 [0.18, 0.84] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 9 | 0.543 [0.22, 0.87] / 0.543 [0.22, 0.87] / 0.551 [0.23, 0.87] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 9 | 0.511 [0.22, 0.84] / 0.646 [0.37, 0.93] / 0.801 [0.54, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.576 [0.32, 0.91] / 0.780 [0.68, 0.93] / 0.780 [0.68, 0.93] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.682 [0.60, 0.80] / 0.822 [0.75, 0.91] / 0.904 [0.86, 0.97] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.570 [0.46, 0.66] / 0.857 [0.81, 0.91] / 0.927 [0.90, 0.96] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.410 [0.11, 0.77] / 0.449 [0.16, 0.80] / 0.767 [0.57, 0.94] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.323 [0.08, 0.75] / 0.324 [0.08, 0.75] / 0.621 [0.32, 0.89] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.078 [0.01, 0.36] / 0.204 [0.06, 0.57] / 0.396 [0.12, 0.72] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 9 | 0.236 [0.01, 0.56] / 0.278 [0.03, 0.59] / 0.414 [0.12, 0.78] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 9 | 0.264 [0.04, 0.71] / 0.629 [0.37, 0.91] / 0.794 [0.52, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.526 [0.26, 0.82] / 0.682 [0.47, 0.91] / 0.808 [0.70, 0.93] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.492 [0.42, 0.59] / 0.874 [0.81, 0.95] / 0.891 [0.85, 0.95] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.547 [0.47, 0.65] / 0.871 [0.83, 0.92] / 0.936 [0.91, 0.97] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.125 [0.00, 0.39] / 0.514 [0.23, 0.81] / 0.806 [0.67, 0.95] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.330 [0.10, 0.75] / 0.565 [0.28, 0.86] / 0.572 [0.29, 0.86] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.185 [0.02, 0.45] / 0.238 [0.06, 0.51] / 0.348 [0.15, 0.66] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 9 | 0.186 [0.02, 0.45] / 0.261 [0.06, 0.53] / 0.322 [0.14, 0.59] |

## AP per class, decoder caps_chain (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 9 | 0.250 [0.04, 0.67] / 0.632 [0.29, 0.96] / 0.866 [0.57, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.671 [0.44, 0.92] / 0.709 [0.52, 0.92] / 0.840 [0.69, 0.96] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.587 [0.43, 0.74] / 0.736 [0.59, 0.86] / 0.823 [0.71, 0.93] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.555 [0.42, 0.68] / 0.828 [0.75, 0.90] / 0.879 [0.81, 0.93] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.136 [0.02, 0.49] / 0.585 [0.29, 0.87] / 0.672 [0.40, 0.91] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.236 [0.02, 0.57] / 0.331 [0.09, 0.68] / 0.415 [0.17, 0.75] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.105 [0.03, 0.27] / 0.105 [0.03, 0.27] / 0.267 [0.11, 0.61] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 9 | 0.348 [0.13, 0.66] / 0.402 [0.20, 0.71] / 0.405 [0.21, 0.71] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 9 | 0.362 [0.07, 0.73] / 0.478 [0.16, 0.86] / 0.767 [0.46, 1.00] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.811 [0.58, 0.98] / 0.901 [0.78, 0.98] / 0.901 [0.78, 0.98] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.688 [0.59, 0.82] / 0.872 [0.78, 0.97] / 0.893 [0.81, 0.97] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.612 [0.50, 0.72] / 0.848 [0.78, 0.90] / 0.879 [0.83, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.322 [0.07, 0.76] / 0.749 [0.45, 1.00] / 0.899 [0.74, 1.00] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.361 [0.06, 0.74] / 0.474 [0.16, 0.85] / 0.589 [0.28, 0.95] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.128 [0.02, 0.44] / 0.293 [0.09, 0.64] / 0.443 [0.18, 0.83] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 9 | 0.561 [0.23, 0.90] / 0.561 [0.23, 0.90] / 0.569 [0.24, 0.90] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 9 | 0.458 [0.16, 0.82] / 0.585 [0.28, 0.91] / 0.800 [0.53, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.567 [0.27, 0.91] / 0.767 [0.62, 0.93] / 0.767 [0.62, 0.93] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.682 [0.60, 0.80] / 0.822 [0.75, 0.91] / 0.904 [0.86, 0.97] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 159 | 0.570 [0.46, 0.66] / 0.857 [0.81, 0.91] / 0.927 [0.90, 0.96] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.431 [0.13, 0.80] / 0.507 [0.18, 0.86] / 0.904 [0.70, 1.00] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.302 [0.07, 0.74] / 0.304 [0.07, 0.74] / 0.598 [0.29, 0.89] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.075 [0.01, 0.36] / 0.199 [0.06, 0.57] / 0.433 [0.15, 0.78] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 9 | 0.236 [0.01, 0.56] / 0.301 [0.04, 0.64] / 0.481 [0.16, 0.87] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 9 | 0.264 [0.04, 0.71] / 0.584 [0.30, 0.90] / 0.794 [0.52, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 18 | 0.533 [0.26, 0.81] / 0.657 [0.41, 0.89] / 0.775 [0.62, 0.92] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 84 | 0.492 [0.42, 0.59] / 0.874 [0.81, 0.95] / 0.891 [0.85, 0.95] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 159 | 0.547 [0.47, 0.65] / 0.871 [0.83, 0.92] / 0.936 [0.91, 0.97] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.133 [0.00, 0.40] / 0.558 [0.23, 0.90] / 0.878 [0.72, 1.00] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.325 [0.09, 0.76] / 0.559 [0.25, 0.87] / 0.566 [0.27, 0.87] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.177 [0.02, 0.47] / 0.228 [0.06, 0.51] / 0.336 [0.13, 0.65] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 9 | 0.196 [0.02, 0.50] / 0.279 [0.07, 0.62] / 0.368 [0.15, 0.73] |

## Phases (held-out)
| model | run | row acc | row macro F1 | F1@50 P0 | F1@50 P1 | acc 1 Hz P0 | acc 1 Hz P1 |
|---|---|---|---|---|---|---|---|
| S L-A3 offline | seed 0 | 0.806 | 0.674 | 54.8 | 66.4 | 0.802 | 0.803 |
| S L-A3 offline | seed 1 | 0.865 | 0.750 | 74.2 | 76.9 | 0.858 | 0.853 |
| S MS-TCN reference | seed 0 | 0.857 | 0.736 | 44.0 | 73.4 | 0.851 | 0.849 |
| S MS-TCN reference | seed 1 | 0.858 | 0.737 | 47.1 | 72.4 | 0.852 | 0.849 |

## Paired Wilcoxon over the 10 held-out videos (L-A3 S, mean of seeds per video, minus the other model)
| key | n | mean diff | median diff | better | worse | p |
|---|---|---|---|---|---|---|
| nms|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.021 | +0.029 | 7 | 3 | 0.557 |
| nms|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.027 | -0.002 | 5 | 5 | 0.846 |
| nms|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.009 | -0.004 | 5 | 5 | 1.000 |
| nms|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.053 | +0.078 | 7 | 3 | 0.232 |
| nms|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.035 | -0.018 | 4 | 6 | 0.846 |
| nms|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.038 | +0.058 | 6 | 4 | 0.826 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.021 | +0.029 | 7 | 3 | 0.557 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.027 | -0.002 | 5 | 5 | 0.846 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.009 | -0.004 | 5 | 5 | 1.000 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.053 | +0.078 | 7 | 3 | 0.232 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.035 | -0.018 | 4 | 6 | 0.846 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.042 | +0.042 | 6 | 4 | 0.826 |
| phases F1@50 P0|S L-A3 offline vs S MS-TCN reference | 10 | +18.951 | +16.816 | 10 | 0 | 0.002 |
| phases F1@50 P1|S L-A3 offline vs S MS-TCN reference | 10 | -1.230 | -1.496 | 4 | 6 | 0.432 |

## Prediction check: long-group mAP@+-2 s, mean of 2 seeds, S L-A3 offline vs reference + 0.03
| decoder | S L-A3 seeds | S L-A3 mean | reference seeds | reference mean | threshold | difference |
|---|---|---|---|---|---|---|
| nms | 0.369, 0.515 | 0.442 | 0.314, 0.395 | 0.354 | 0.384 | +0.088 |
| caps_chain | 0.356, 0.519 | 0.438 | 0.328, 0.406 | 0.367 | 0.397 | +0.071 |
