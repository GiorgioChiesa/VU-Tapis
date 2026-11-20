# EXP-22 fold 3: L-A3 on EXP-19 S latents vs reference (held-out, 10 videos, 95% CI = per-video bootstrap)

## Group mAP, decoder nms (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.582 [0.46, 0.72] / 0.770 [0.67, 0.88] / 0.828 [0.72, 0.93] | 0.094 [0.04, 0.23] / 0.445 [0.35, 0.61] / 0.526 [0.44, 0.68] |
| S L-A3 offline | seed 1 | 0.550 [0.46, 0.66] / 0.721 [0.60, 0.86] / 0.858 [0.76, 0.94] | 0.188 [0.14, 0.35] / 0.406 [0.29, 0.62] / 0.589 [0.46, 0.78] |
| S L-A3 offline | mean of seeds | 0.566 / 0.745 / 0.843 | 0.141 / 0.425 / 0.557 |
| S MS-TCN reference | seed 0 | 0.515 [0.40, 0.67] / 0.828 [0.73, 0.92] / 0.894 [0.82, 0.96] | 0.184 [0.11, 0.34] / 0.302 [0.20, 0.49] / 0.559 [0.43, 0.73] |
| S MS-TCN reference | seed 1 | 0.492 [0.34, 0.66] / 0.839 [0.75, 0.92] / 0.890 [0.81, 0.95] | 0.184 [0.08, 0.36] / 0.292 [0.16, 0.50] / 0.514 [0.42, 0.67] |
| S MS-TCN reference | mean of seeds | 0.504 / 0.833 / 0.892 | 0.184 / 0.297 / 0.536 |

## Group mAP, decoder caps_chain (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.589 [0.47, 0.72] / 0.778 [0.68, 0.88] / 0.842 [0.74, 0.94] | 0.089 [0.04, 0.22] / 0.426 [0.32, 0.59] / 0.509 [0.43, 0.66] |
| S L-A3 offline | seed 1 | 0.551 [0.46, 0.66] / 0.724 [0.60, 0.87] / 0.861 [0.77, 0.94] | 0.185 [0.13, 0.35] / 0.390 [0.28, 0.62] / 0.579 [0.45, 0.78] |
| S L-A3 offline | mean of seeds | 0.570 / 0.751 / 0.852 | 0.137 / 0.408 / 0.544 |
| S MS-TCN reference | seed 0 | 0.514 [0.40, 0.67] / 0.828 [0.73, 0.92] / 0.893 [0.82, 0.96] | 0.199 [0.11, 0.36] / 0.328 [0.21, 0.52] / 0.580 [0.43, 0.77] |
| S MS-TCN reference | seed 1 | 0.498 [0.35, 0.66] / 0.842 [0.75, 0.92] / 0.894 [0.81, 0.96] | 0.188 [0.08, 0.36] / 0.271 [0.14, 0.48] / 0.522 [0.42, 0.69] |
| S MS-TCN reference | mean of seeds | 0.506 / 0.835 / 0.894 | 0.193 / 0.300 / 0.551 |

## AP per class, decoder nms (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 10 | 0.557 [0.25, 0.90] / 0.700 [0.40, 0.90] / 0.801 [0.50, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.411 [0.23, 0.69] / 0.610 [0.41, 0.82] / 0.683 [0.51, 0.87] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.737 [0.64, 0.83] / 0.916 [0.88, 0.95] / 0.935 [0.89, 0.97] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 158 | 0.622 [0.51, 0.74] / 0.853 [0.77, 0.93] / 0.892 [0.82, 0.95] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.053 [0.01, 0.20] / 0.331 [0.16, 0.63] / 0.331 [0.16, 0.63] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.151 [0.00, 0.55] / 0.797 [0.57, 1.00] / 0.892 [0.67, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.139 [0.02, 0.46] / 0.425 [0.25, 0.76] / 0.621 [0.50, 0.87] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 10 | 0.033 [0.00, 0.12] / 0.227 [0.07, 0.48] / 0.261 [0.11, 0.50] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 10 | 0.273 [0.02, 0.64] / 0.530 [0.20, 0.90] / 0.806 [0.51, 1.00] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.543 [0.41, 0.70] / 0.627 [0.47, 0.82] / 0.815 [0.69, 0.93] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.797 [0.72, 0.86] / 0.922 [0.88, 0.95] / 0.954 [0.93, 0.97] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 158 | 0.588 [0.49, 0.72] / 0.805 [0.74, 0.88] / 0.857 [0.78, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.228 [0.03, 0.56] / 0.375 [0.12, 0.71] / 0.380 [0.13, 0.71] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.222 [0.02, 0.68] / 0.449 [0.13, 0.89] / 0.906 [0.71, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.245 [0.10, 0.55] / 0.327 [0.16, 0.66] / 0.592 [0.47, 0.88] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 10 | 0.054 [0.00, 0.31] / 0.471 [0.26, 0.75] / 0.476 [0.26, 0.75] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 10 | 0.368 [0.08, 0.79] / 0.701 [0.40, 1.00] / 0.801 [0.50, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.602 [0.36, 0.89] / 0.926 [0.85, 0.99] / 0.926 [0.85, 0.99] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.613 [0.50, 0.72] / 0.875 [0.83, 0.92] / 0.945 [0.91, 0.98] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 158 | 0.477 [0.39, 0.58] / 0.810 [0.74, 0.89] / 0.903 [0.85, 0.95] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.072 [0.01, 0.27] / 0.161 [0.04, 0.46] / 0.383 [0.20, 0.70] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.339 [0.06, 0.75] / 0.514 [0.19, 0.92] / 0.733 [0.47, 1.00] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.022 [0.00, 0.13] / 0.106 [0.02, 0.30] / 0.686 [0.45, 0.93] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 10 | 0.303 [0.05, 0.64] / 0.429 [0.18, 0.71] / 0.432 [0.19, 0.71] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 10 | 0.510 [0.20, 0.90] / 0.789 [0.50, 1.00] / 0.805 [0.54, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.555 [0.30, 0.80] / 0.896 [0.81, 0.97] / 0.896 [0.81, 0.97] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.423 [0.30, 0.54] / 0.867 [0.81, 0.92] / 0.946 [0.90, 0.98] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 158 | 0.481 [0.38, 0.61] / 0.804 [0.72, 0.89] / 0.914 [0.87, 0.95] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.189 [0.05, 0.45] / 0.213 [0.08, 0.48] / 0.278 [0.13, 0.57] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.243 [0.03, 0.63] / 0.338 [0.06, 0.80] / 0.746 [0.48, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.040 [0.00, 0.21] / 0.263 [0.08, 0.53] / 0.661 [0.51, 0.89] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 10 | 0.263 [0.04, 0.53] / 0.354 [0.11, 0.59] / 0.371 [0.14, 0.60] |

## AP per class, decoder caps_chain (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 10 | 0.557 [0.25, 0.90] / 0.700 [0.40, 0.90] / 0.801 [0.50, 1.00] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.439 [0.25, 0.70] / 0.644 [0.45, 0.86] / 0.741 [0.58, 0.91] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.737 [0.64, 0.83] / 0.916 [0.88, 0.95] / 0.935 [0.89, 0.97] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 158 | 0.622 [0.51, 0.74] / 0.853 [0.77, 0.93] / 0.892 [0.82, 0.95] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.040 [0.01, 0.14] / 0.258 [0.11, 0.58] / 0.258 [0.11, 0.58] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.151 [0.00, 0.55] / 0.782 [0.52, 1.00] / 0.892 [0.67, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.133 [0.02, 0.45] / 0.452 [0.25, 0.80] / 0.637 [0.50, 0.92] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 10 | 0.032 [0.00, 0.11] / 0.214 [0.07, 0.45] / 0.247 [0.11, 0.48] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 10 | 0.273 [0.02, 0.64] / 0.530 [0.20, 0.90] / 0.806 [0.51, 1.00] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.547 [0.43, 0.69] / 0.638 [0.48, 0.83] / 0.828 [0.71, 0.93] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.797 [0.72, 0.86] / 0.922 [0.88, 0.95] / 0.954 [0.93, 0.97] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 158 | 0.588 [0.49, 0.72] / 0.805 [0.74, 0.88] / 0.857 [0.78, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.217 [0.02, 0.55] / 0.359 [0.11, 0.70] / 0.364 [0.12, 0.70] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.222 [0.02, 0.68] / 0.444 [0.12, 0.88] / 0.906 [0.71, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.241 [0.09, 0.55] / 0.314 [0.16, 0.64] / 0.599 [0.45, 0.90] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 10 | 0.059 [0.00, 0.32] / 0.444 [0.23, 0.74] / 0.449 [0.24, 0.74] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 10 | 0.368 [0.08, 0.79] / 0.701 [0.40, 1.00] / 0.801 [0.50, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.599 [0.35, 0.89] / 0.925 [0.84, 0.99] / 0.925 [0.84, 0.99] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.613 [0.50, 0.72] / 0.875 [0.83, 0.92] / 0.945 [0.91, 0.98] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 158 | 0.477 [0.39, 0.58] / 0.810 [0.74, 0.89] / 0.903 [0.85, 0.95] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.079 [0.01, 0.30] / 0.175 [0.04, 0.49] / 0.395 [0.20, 0.72] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.339 [0.06, 0.75] / 0.536 [0.20, 1.00] / 0.760 [0.49, 1.00] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.018 [0.00, 0.11] / 0.098 [0.02, 0.27] / 0.659 [0.43, 0.93] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 10 | 0.358 [0.07, 0.68] / 0.501 [0.22, 0.81] / 0.505 [0.23, 0.81] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 10 | 0.525 [0.20, 0.90] / 0.800 [0.50, 1.00] / 0.815 [0.53, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 20 | 0.564 [0.32, 0.80] / 0.899 [0.79, 0.98] / 0.899 [0.79, 0.98] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 112 | 0.423 [0.30, 0.54] / 0.867 [0.81, 0.92] / 0.946 [0.90, 0.98] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 158 | 0.481 [0.38, 0.61] / 0.804 [0.72, 0.89] / 0.914 [0.87, 0.95] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.189 [0.05, 0.45] / 0.213 [0.08, 0.48] / 0.288 [0.14, 0.62] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.268 [0.03, 0.68] / 0.368 [0.07, 0.83] / 0.763 [0.49, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.035 [0.00, 0.21] / 0.205 [0.06, 0.48] / 0.718 [0.48, 1.00] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 10 | 0.259 [0.04, 0.52] / 0.300 [0.08, 0.56] / 0.317 [0.10, 0.58] |

## Phases (held-out)
| model | run | row acc | row macro F1 | F1@50 P0 | F1@50 P1 | acc 1 Hz P0 | acc 1 Hz P1 |
|---|---|---|---|---|---|---|---|
| S L-A3 offline | seed 0 | 0.814 | 0.688 | 54.5 | 69.5 | 0.810 | 0.808 |
| S L-A3 offline | seed 1 | 0.911 | 0.788 | 72.0 | 80.4 | 0.910 | 0.911 |
| S MS-TCN reference | seed 0 | 0.876 | 0.757 | 45.1 | 75.5 | 0.878 | 0.878 |
| S MS-TCN reference | seed 1 | 0.871 | 0.751 | 36.2 | 73.1 | 0.872 | 0.873 |

## Paired Wilcoxon over the 10 held-out videos (L-A3 S, mean of seeds per video, minus the other model)
| key | n | mean diff | median diff | better | worse | p |
|---|---|---|---|---|---|---|
| nms|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.059 | +0.081 | 7 | 3 | 0.049 |
| nms|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.065 | -0.047 | 1 | 9 | 0.014 |
| nms|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.035 | -0.038 | 2 | 8 | 0.105 |
| nms|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | -0.037 | -0.008 | 4 | 5 | 0.734 |
| nms|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.089 | +0.072 | 7 | 3 | 0.160 |
| nms|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.011 | +0.026 | 6 | 4 | 0.846 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.059 | +0.081 | 7 | 3 | 0.049 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.065 | -0.047 | 1 | 9 | 0.014 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.035 | -0.038 | 2 | 8 | 0.105 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | -0.037 | -0.008 | 4 | 5 | 0.734 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.089 | +0.072 | 7 | 3 | 0.160 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.011 | +0.026 | 6 | 4 | 0.846 |
| phases F1@50 P0|S L-A3 offline vs S MS-TCN reference | 10 | +22.653 | +25.838 | 9 | 1 | 0.010 |
| phases F1@50 P1|S L-A3 offline vs S MS-TCN reference | 10 | +0.623 | -0.447 | 4 | 5 | 0.910 |

## Prediction check: long-group mAP@+-2 s, mean of 2 seeds, S L-A3 offline vs reference + 0.03
| decoder | S L-A3 seeds | S L-A3 mean | reference seeds | reference mean | threshold | difference |
|---|---|---|---|---|---|---|
| nms | 0.445, 0.406 | 0.425 | 0.302, 0.292 | 0.297 | 0.327 | +0.128 |
| caps_chain | 0.426, 0.390 | 0.408 | 0.328, 0.271 | 0.300 | 0.330 | +0.109 |
