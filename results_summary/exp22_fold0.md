# EXP-22 fold 0: L-A3 on EXP-19 S latents vs reference (held-out, 10 videos, 95% CI = per-video bootstrap)

## Group mAP, decoder nms (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.589 [0.48, 0.70] / 0.724 [0.60, 0.85] / 0.752 [0.62, 0.88] | 0.192 [0.07, 0.42] / 0.419 [0.31, 0.63] / 0.607 [0.53, 0.75] |
| S L-A3 offline | seed 1 | 0.543 [0.45, 0.65] / 0.692 [0.56, 0.84] / 0.789 [0.64, 0.90] | 0.178 [0.10, 0.34] / 0.357 [0.29, 0.51] / 0.532 [0.43, 0.71] |
| S L-A3 offline | mean of seeds | 0.566 / 0.708 / 0.771 | 0.185 / 0.388 / 0.569 |
| S MS-TCN reference | seed 0 | 0.532 [0.41, 0.69] / 0.722 [0.60, 0.85] / 0.848 [0.74, 0.93] | 0.242 [0.16, 0.39] / 0.429 [0.33, 0.63] / 0.611 [0.51, 0.79] |
| S MS-TCN reference | seed 1 | 0.529 [0.45, 0.62] / 0.811 [0.67, 0.92] / 0.890 [0.80, 0.95] | 0.188 [0.09, 0.39] / 0.297 [0.18, 0.52] / 0.591 [0.46, 0.78] |
| S MS-TCN reference | mean of seeds | 0.531 / 0.766 / 0.869 | 0.215 / 0.363 / 0.601 |
| M L-A3 offline (EXP-21, long head) | seed 0 | - / - / - | 0.160 [0.09, 0.34] / 0.357 [0.22, 0.58] / 0.501 [0.41, 0.66] |
| M L-A3 offline (EXP-21, long head) | seed 1 | - / - / - | 0.098 [0.04, 0.27] / 0.303 [0.20, 0.50] / 0.499 [0.42, 0.66] |
| M L-A3 offline (EXP-21, long head) | mean of seeds | - / - / - | 0.129 / 0.330 / 0.500 |
| M short MS-TCN (EXP-21 p3) | seed 0 | 0.205 [0.16, 0.29] / 0.567 [0.43, 0.71] / 0.695 [0.56, 0.82] | 0.151 [0.12, 0.26] / 0.249 [0.18, 0.38] / 0.439 [0.34, 0.57] |
| M short MS-TCN (EXP-21 p3) | seed 1 | 0.319 [0.25, 0.45] / 0.634 [0.48, 0.79] / 0.748 [0.63, 0.86] | 0.133 [0.07, 0.24] / 0.258 [0.19, 0.38] / 0.407 [0.35, 0.53] |
| M short MS-TCN (EXP-21 p3) | mean of seeds | 0.262 / 0.601 / 0.722 | 0.142 / 0.253 / 0.423 |

## Group mAP, decoder caps_chain (mAP@1 / @2 / @5)
| model | run | short | long |
|---|---|---|---|
| S L-A3 offline | seed 0 | 0.586 [0.48, 0.70] / 0.720 [0.58, 0.85] / 0.751 [0.62, 0.88] | 0.215 [0.08, 0.44] / 0.437 [0.32, 0.65] / 0.648 [0.56, 0.78] |
| S L-A3 offline | seed 1 | 0.544 [0.45, 0.65] / 0.693 [0.56, 0.84] / 0.790 [0.65, 0.90] | 0.169 [0.09, 0.33] / 0.350 [0.28, 0.50] / 0.532 [0.43, 0.71] |
| S L-A3 offline | mean of seeds | 0.565 / 0.706 / 0.770 | 0.192 / 0.393 / 0.590 |
| S MS-TCN reference | seed 0 | 0.551 [0.42, 0.72] / 0.738 [0.61, 0.87] / 0.871 [0.75, 0.96] | 0.231 [0.15, 0.39] / 0.433 [0.33, 0.65] / 0.627 [0.52, 0.81] |
| S MS-TCN reference | seed 1 | 0.530 [0.45, 0.62] / 0.820 [0.68, 0.93] / 0.905 [0.82, 0.96] | 0.193 [0.09, 0.40] / 0.299 [0.18, 0.52] / 0.592 [0.45, 0.78] |
| S MS-TCN reference | mean of seeds | 0.541 / 0.779 / 0.888 | 0.212 / 0.366 / 0.609 |
| M L-A3 offline (EXP-21, long head) | seed 0 | - / - / - | 0.166 [0.09, 0.33] / 0.367 [0.23, 0.59] / 0.511 [0.41, 0.66] |
| M L-A3 offline (EXP-21, long head) | seed 1 | - / - / - | 0.097 [0.04, 0.28] / 0.299 [0.19, 0.50] / 0.511 [0.43, 0.67] |
| M L-A3 offline (EXP-21, long head) | mean of seeds | - / - / - | 0.131 / 0.333 / 0.511 |
| M short MS-TCN (EXP-21 p3) | seed 0 | 0.210 [0.16, 0.30] / 0.593 [0.44, 0.75] / 0.728 [0.57, 0.87] | 0.169 [0.13, 0.31] / 0.271 [0.20, 0.43] / 0.457 [0.34, 0.63] |
| M short MS-TCN (EXP-21 p3) | seed 1 | 0.322 [0.25, 0.45] / 0.642 [0.49, 0.80] / 0.767 [0.63, 0.88] | 0.140 [0.07, 0.26] / 0.275 [0.19, 0.41] / 0.457 [0.40, 0.57] |
| M short MS-TCN (EXP-21 p3) | mean of seeds | 0.266 / 0.617 / 0.748 | 0.154 / 0.273 / 0.457 |

## AP per class, decoder nms (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 9 | 0.157 [0.00, 0.46] / 0.465 [0.18, 0.78] / 0.561 [0.23, 0.89] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.715 [0.47, 0.94] / 0.774 [0.54, 0.95] / 0.777 [0.54, 0.95] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.738 [0.51, 0.89] / 0.813 [0.60, 0.93] / 0.813 [0.60, 0.93] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 123 | 0.745 [0.64, 0.85] / 0.842 [0.77, 0.91] / 0.858 [0.79, 0.93] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.166 [0.01, 0.51] / 0.288 [0.06, 0.69] / 0.600 [0.29, 0.90] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.303 [0.07, 0.64] / 0.478 [0.23, 0.77] / 0.639 [0.32, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.049 [0.00, 0.25] / 0.301 [0.13, 0.67] / 0.508 [0.29, 0.85] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 10 | 0.252 [0.03, 0.70] / 0.608 [0.31, 0.90] / 0.681 [0.42, 0.93] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 9 | 0.074 [0.00, 0.35] / 0.335 [0.06, 0.72] / 0.669 [0.34, 0.90] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.694 [0.43, 0.93] / 0.778 [0.58, 0.95] / 0.779 [0.58, 0.95] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.653 [0.46, 0.81] / 0.811 [0.56, 0.96] / 0.847 [0.62, 0.96] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 123 | 0.751 [0.63, 0.84] / 0.843 [0.78, 0.90] / 0.860 [0.79, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.351 [0.13, 0.67] / 0.399 [0.18, 0.71] / 0.503 [0.28, 0.80] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.040 [0.00, 0.26] / 0.370 [0.08, 0.81] / 0.636 [0.27, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.241 [0.08, 0.56] / 0.435 [0.25, 0.80] / 0.445 [0.26, 0.82] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 10 | 0.080 [0.00, 0.28] / 0.226 [0.08, 0.52] / 0.543 [0.31, 0.84] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 9 | 0.314 [0.06, 0.70] / 0.330 [0.07, 0.71] / 0.740 [0.54, 0.93] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.487 [0.22, 0.77] / 0.844 [0.64, 0.97] / 0.845 [0.64, 0.97] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.621 [0.49, 0.73] / 0.806 [0.65, 0.93] / 0.872 [0.74, 0.95] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 123 | 0.707 [0.61, 0.79] / 0.907 [0.86, 0.95] / 0.936 [0.90, 0.97] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.317 [0.08, 0.63] / 0.454 [0.24, 0.74] / 0.768 [0.55, 0.93] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.367 [0.08, 0.72] / 0.618 [0.36, 0.91] / 0.703 [0.43, 0.97] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.091 [0.00, 0.39] / 0.252 [0.09, 0.62] / 0.531 [0.38, 0.85] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 10 | 0.193 [0.00, 0.52] / 0.391 [0.12, 0.78] / 0.441 [0.19, 0.80] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 9 | 0.391 [0.09, 0.75] / 0.781 [0.52, 1.00] / 0.944 [0.86, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.528 [0.23, 0.81] / 0.706 [0.48, 0.92] / 0.783 [0.60, 0.94] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.512 [0.38, 0.65] / 0.848 [0.69, 0.94] / 0.894 [0.76, 0.96] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 123 | 0.687 [0.61, 0.79] / 0.908 [0.86, 0.96] / 0.938 [0.90, 0.97] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.137 [0.01, 0.47] / 0.323 [0.17, 0.62] / 0.702 [0.47, 0.91] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.305 [0.07, 0.66] / 0.455 [0.20, 0.83] / 0.753 [0.47, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.242 [0.05, 0.62] / 0.309 [0.12, 0.68] / 0.506 [0.38, 0.80] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 10 | 0.069 [0.00, 0.31] / 0.100 [0.02, 0.35] / 0.402 [0.12, 0.72] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Start_dissection | 10 | 0.180 [0.03, 0.52] / 0.495 [0.25, 0.82] / 0.719 [0.49, 0.92] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.115 [0.00, 0.38] / 0.370 [0.15, 0.75] / 0.605 [0.34, 0.86] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Transection_of_the_urethra | 10 | 0.306 [0.07, 0.71] / 0.408 [0.17, 0.80] / 0.515 [0.32, 0.82] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Stitch_in_bladder | 10 | 0.039 [0.01, 0.15] / 0.155 [0.02, 0.37] / 0.163 [0.03, 0.37] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Start_dissection | 10 | 0.018 [0.00, 0.15] / 0.454 [0.18, 0.81] / 0.802 [0.59, 0.97] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.095 [0.00, 0.40] / 0.366 [0.13, 0.76] / 0.653 [0.36, 0.92] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Transection_of_the_urethra | 10 | 0.200 [0.07, 0.52] / 0.247 [0.12, 0.55] / 0.387 [0.20, 0.71] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Stitch_in_bladder | 10 | 0.079 [0.02, 0.27] / 0.147 [0.04, 0.43] / 0.154 [0.05, 0.43] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Grasping_catheter_tip | 9 | 0.057 [0.00, 0.22] / 0.335 [0.08, 0.65] / 0.645 [0.40, 0.87] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.147 [0.04, 0.32] / 0.485 [0.24, 0.75] / 0.514 [0.27, 0.78] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.308 [0.20, 0.48] / 0.658 [0.50, 0.82] / 0.773 [0.62, 0.88] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Metal_clip_on_right_pedicle | 123 | 0.306 [0.23, 0.38] / 0.790 [0.71, 0.85] / 0.850 [0.78, 0.91] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Start_dissection | 10 | 0.214 [0.03, 0.60] / 0.492 [0.23, 0.82] / 0.605 [0.36, 0.88] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.110 [0.01, 0.42] / 0.222 [0.03, 0.59] / 0.629 [0.42, 0.85] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Transection_of_the_urethra | 10 | 0.250 [0.11, 0.49] / 0.250 [0.11, 0.49] / 0.465 [0.32, 0.69] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Stitch_in_bladder | 10 | 0.028 [0.01, 0.10] / 0.032 [0.01, 0.11] / 0.057 [0.03, 0.13] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Grasping_catheter_tip | 9 | 0.185 [0.02, 0.56] / 0.457 [0.20, 0.81] / 0.782 [0.57, 0.98] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.272 [0.03, 0.59] / 0.550 [0.28, 0.81] / 0.582 [0.34, 0.83] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.363 [0.26, 0.49] / 0.715 [0.53, 0.84] / 0.783 [0.62, 0.89] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Metal_clip_on_right_pedicle | 123 | 0.457 [0.36, 0.55] / 0.814 [0.74, 0.87] / 0.845 [0.76, 0.91] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Start_dissection | 10 | 0.272 [0.04, 0.60] / 0.427 [0.16, 0.77] / 0.568 [0.38, 0.82] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.144 [0.00, 0.39] / 0.321 [0.09, 0.60] / 0.542 [0.38, 0.75] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Transection_of_the_urethra | 10 | 0.102 [0.01, 0.30] / 0.241 [0.10, 0.48] / 0.466 [0.30, 0.73] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Stitch_in_bladder | 10 | 0.013 [0.00, 0.06] / 0.042 [0.02, 0.14] / 0.052 [0.02, 0.15] |

## AP per class, decoder caps_chain (AP@1 / @2 / @5)
| model | run | class | n_gt | AP |
|---|---|---|---|---|
| S L-A3 offline | seed 0 | Grasping_catheter_tip | 9 | 0.157 [0.00, 0.46] / 0.454 [0.16, 0.78] / 0.561 [0.23, 0.89] |
| S L-A3 offline | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.705 [0.46, 0.95] / 0.769 [0.52, 0.96] / 0.772 [0.52, 0.96] |
| S L-A3 offline | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.738 [0.51, 0.89] / 0.813 [0.60, 0.93] / 0.813 [0.60, 0.93] |
| S L-A3 offline | seed 0 | Metal_clip_on_right_pedicle | 123 | 0.745 [0.64, 0.85] / 0.842 [0.77, 0.91] / 0.858 [0.79, 0.93] |
| S L-A3 offline | seed 0 | Start_dissection | 10 | 0.201 [0.01, 0.55] / 0.327 [0.07, 0.71] / 0.634 [0.32, 0.90] |
| S L-A3 offline | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.288 [0.06, 0.64] / 0.453 [0.20, 0.78] / 0.651 [0.31, 1.00] |
| S L-A3 offline | seed 0 | Transection_of_the_urethra | 10 | 0.090 [0.01, 0.36] / 0.347 [0.15, 0.74] / 0.580 [0.32, 0.91] |
| S L-A3 offline | seed 0 | Stitch_in_bladder | 10 | 0.281 [0.03, 0.70] / 0.622 [0.31, 0.90] / 0.725 [0.43, 1.00] |
| S L-A3 offline | seed 1 | Grasping_catheter_tip | 9 | 0.074 [0.00, 0.35] / 0.335 [0.06, 0.72] / 0.669 [0.34, 0.90] |
| S L-A3 offline | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.697 [0.43, 0.93] / 0.782 [0.57, 0.95] / 0.783 [0.57, 0.95] |
| S L-A3 offline | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.653 [0.46, 0.81] / 0.811 [0.56, 0.96] / 0.847 [0.62, 0.96] |
| S L-A3 offline | seed 1 | Metal_clip_on_right_pedicle | 123 | 0.751 [0.63, 0.84] / 0.843 [0.78, 0.90] / 0.860 [0.79, 0.92] |
| S L-A3 offline | seed 1 | Start_dissection | 10 | 0.336 [0.12, 0.66] / 0.384 [0.16, 0.70] / 0.492 [0.26, 0.80] |
| S L-A3 offline | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.040 [0.00, 0.26] / 0.382 [0.08, 0.81] / 0.653 [0.29, 1.00] |
| S L-A3 offline | seed 1 | Transection_of_the_urethra | 10 | 0.241 [0.08, 0.56] / 0.435 [0.25, 0.80] / 0.480 [0.27, 0.85] |
| S L-A3 offline | seed 1 | Stitch_in_bladder | 10 | 0.060 [0.00, 0.23] / 0.198 [0.07, 0.48] / 0.502 [0.28, 0.82] |
| S MS-TCN reference | seed 0 | Grasping_catheter_tip | 9 | 0.389 [0.08, 0.82] / 0.405 [0.11, 0.82] / 0.837 [0.60, 1.00] |
| S MS-TCN reference | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.486 [0.22, 0.77] / 0.836 [0.63, 0.97] / 0.837 [0.63, 0.97] |
| S MS-TCN reference | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.621 [0.49, 0.73] / 0.806 [0.65, 0.93] / 0.872 [0.74, 0.95] |
| S MS-TCN reference | seed 0 | Metal_clip_on_right_pedicle | 123 | 0.707 [0.61, 0.79] / 0.907 [0.86, 0.95] / 0.936 [0.90, 0.97] |
| S MS-TCN reference | seed 0 | Start_dissection | 10 | 0.303 [0.07, 0.62] / 0.434 [0.23, 0.72] / 0.774 [0.55, 0.96] |
| S MS-TCN reference | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.337 [0.06, 0.69] / 0.637 [0.37, 0.94] / 0.732 [0.46, 1.00] |
| S MS-TCN reference | seed 0 | Transection_of_the_urethra | 10 | 0.091 [0.00, 0.39] / 0.272 [0.10, 0.67] / 0.567 [0.39, 0.92] |
| S MS-TCN reference | seed 0 | Stitch_in_bladder | 10 | 0.193 [0.00, 0.52] / 0.387 [0.11, 0.78] / 0.435 [0.16, 0.81] |
| S MS-TCN reference | seed 1 | Grasping_catheter_tip | 9 | 0.391 [0.09, 0.75] / 0.796 [0.53, 1.00] / 0.978 [0.90, 1.00] |
| S MS-TCN reference | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.532 [0.23, 0.81] / 0.729 [0.50, 0.94] / 0.811 [0.63, 0.96] |
| S MS-TCN reference | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.512 [0.38, 0.65] / 0.848 [0.69, 0.94] / 0.894 [0.76, 0.96] |
| S MS-TCN reference | seed 1 | Metal_clip_on_right_pedicle | 123 | 0.687 [0.61, 0.79] / 0.908 [0.86, 0.96] / 0.938 [0.90, 0.97] |
| S MS-TCN reference | seed 1 | Start_dissection | 10 | 0.133 [0.01, 0.47] / 0.310 [0.17, 0.60] / 0.707 [0.46, 0.93] |
| S MS-TCN reference | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.329 [0.07, 0.69] / 0.485 [0.20, 0.87] / 0.784 [0.47, 1.00] |
| S MS-TCN reference | seed 1 | Transection_of_the_urethra | 10 | 0.240 [0.05, 0.62] / 0.307 [0.12, 0.68] / 0.482 [0.38, 0.79] |
| S MS-TCN reference | seed 1 | Stitch_in_bladder | 10 | 0.069 [0.00, 0.31] / 0.094 [0.02, 0.34] / 0.393 [0.11, 0.72] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Start_dissection | 10 | 0.174 [0.03, 0.54] / 0.493 [0.24, 0.83] / 0.718 [0.46, 0.94] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.108 [0.00, 0.40] / 0.363 [0.12, 0.77] / 0.601 [0.29, 0.90] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Transection_of_the_urethra | 10 | 0.342 [0.07, 0.76] / 0.460 [0.17, 0.83] / 0.567 [0.34, 0.88] |
| M L-A3 offline (EXP-21, long head) | seed 0 | Stitch_in_bladder | 10 | 0.039 [0.01, 0.13] / 0.151 [0.02, 0.36] / 0.159 [0.03, 0.37] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Start_dissection | 10 | 0.018 [0.00, 0.15] / 0.445 [0.17, 0.80] / 0.803 [0.57, 1.00] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.087 [0.00, 0.38] / 0.356 [0.13, 0.75] / 0.695 [0.38, 0.94] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Transection_of_the_urethra | 10 | 0.214 [0.07, 0.56] / 0.261 [0.12, 0.60] / 0.403 [0.20, 0.74] |
| M L-A3 offline (EXP-21, long head) | seed 1 | Stitch_in_bladder | 10 | 0.067 [0.02, 0.26] / 0.134 [0.04, 0.40] / 0.141 [0.05, 0.40] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Grasping_catheter_tip | 9 | 0.064 [0.00, 0.28] / 0.391 [0.09, 0.78] / 0.730 [0.44, 1.00] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.160 [0.04, 0.35] / 0.533 [0.25, 0.80] / 0.561 [0.29, 0.83] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.308 [0.20, 0.48] / 0.658 [0.50, 0.82] / 0.773 [0.62, 0.88] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Metal_clip_on_right_pedicle | 123 | 0.306 [0.23, 0.38] / 0.790 [0.71, 0.85] / 0.850 [0.78, 0.91] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Start_dissection | 10 | 0.214 [0.03, 0.60] / 0.492 [0.23, 0.82] / 0.581 [0.32, 0.87] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Incision_of_Denonvilliers_fascia | 9 | 0.158 [0.02, 0.53] / 0.284 [0.05, 0.72] / 0.710 [0.45, 0.95] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Transection_of_the_urethra | 10 | 0.270 [0.11, 0.55] / 0.270 [0.11, 0.55] / 0.478 [0.32, 0.78] |
| M short MS-TCN (EXP-21 p3) | seed 0 | Stitch_in_bladder | 10 | 0.034 [0.01, 0.16] / 0.037 [0.01, 0.16] / 0.060 [0.03, 0.17] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Grasping_catheter_tip | 9 | 0.180 [0.02, 0.55] / 0.442 [0.18, 0.79] / 0.812 [0.59, 1.00] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Clip_or_coagulate_vas_deferens_-_right | 17 | 0.290 [0.04, 0.61] / 0.597 [0.33, 0.86] / 0.627 [0.37, 0.87] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Identification_and_clipping_of_SV_arteries_-_right | 50 | 0.363 [0.26, 0.49] / 0.715 [0.53, 0.84] / 0.783 [0.62, 0.89] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Metal_clip_on_right_pedicle | 123 | 0.457 [0.36, 0.55] / 0.814 [0.74, 0.87] / 0.845 [0.76, 0.91] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Start_dissection | 10 | 0.295 [0.05, 0.66] / 0.453 [0.17, 0.80] / 0.579 [0.37, 0.85] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Incision_of_Denonvilliers_fascia | 9 | 0.162 [0.01, 0.45] / 0.396 [0.11, 0.73] / 0.693 [0.42, 1.00] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Transection_of_the_urethra | 10 | 0.088 [0.01, 0.28] / 0.214 [0.08, 0.46] / 0.508 [0.30, 0.80] |
| M short MS-TCN (EXP-21 p3) | seed 1 | Stitch_in_bladder | 10 | 0.013 [0.00, 0.06] / 0.036 [0.02, 0.11] / 0.046 [0.02, 0.13] |

## Phases (held-out)
| model | run | row acc | row macro F1 | F1@50 P0 | F1@50 P1 | acc 1 Hz P0 | acc 1 Hz P1 |
|---|---|---|---|---|---|---|---|
| S L-A3 offline | seed 0 | 0.863 | 0.747 | 75.5 | 79.6 | 0.857 | 0.858 |
| S L-A3 offline | seed 1 | 0.855 | 0.740 | 69.0 | 77.6 | 0.850 | 0.851 |
| S MS-TCN reference | seed 0 | 0.835 | 0.713 | 41.3 | 71.3 | 0.831 | 0.833 |
| S MS-TCN reference | seed 1 | 0.832 | 0.710 | 40.3 | 70.4 | 0.829 | 0.830 |
| M L-A3 offline (EXP-21, long head) | seed 0 | 0.768 | 0.618 | 51.9 | 71.0 | 0.762 | 0.764 |
| M L-A3 offline (EXP-21, long head) | seed 1 | 0.788 | 0.665 | 58.8 | 69.2 | 0.784 | 0.784 |
| M short MS-TCN (EXP-21 p3) | seed 0 | 0.721 | 0.565 | 20.1 | 51.0 | 0.717 | 0.717 |
| M short MS-TCN (EXP-21 p3) | seed 1 | 0.727 | 0.574 | 20.1 | 55.3 | 0.723 | 0.728 |

## Paired Wilcoxon over the 10 held-out videos (L-A3 S, mean of seeds per video, minus the other model)
| key | n | mean diff | median diff | better | worse | p |
|---|---|---|---|---|---|---|
| nms|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.008 | +0.039 | 7 | 3 | 0.770 |
| nms|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.070 | -0.038 | 2 | 8 | 0.020 |
| nms|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.113 | -0.073 | 0 | 10 | 0.002 |
| nms|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.001 | +0.022 | 5 | 5 | 0.748 |
| nms|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.030 | +0.063 | 6 | 3 | 0.359 |
| nms|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.014 | -0.033 | 4 | 6 | 0.625 |
| nms|S L-A3 offline vs M L-A3 offline (EXP-21, long head)|long|1 | 10 | +0.051 | -0.004 | 4 | 6 | 0.922 |
| nms|S L-A3 offline vs M L-A3 offline (EXP-21, long head)|long|2 | 10 | +0.060 | +0.059 | 5 | 4 | 0.359 |
| nms|S L-A3 offline vs M L-A3 offline (EXP-21, long head)|long|5 | 10 | +0.098 | +0.126 | 7 | 3 | 0.131 |
| nms|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|short|1 | 10 | +0.237 | +0.249 | 10 | 0 | 0.002 |
| nms|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|short|2 | 10 | +0.043 | +0.012 | 5 | 5 | 0.770 |
| nms|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|short|5 | 10 | -0.022 | -0.011 | 4 | 6 | 0.557 |
| nms|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|long|1 | 10 | +0.031 | +0.069 | 6 | 4 | 0.922 |
| nms|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|long|2 | 10 | +0.112 | +0.078 | 8 | 2 | 0.027 |
| nms|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|long|5 | 10 | +0.115 | +0.088 | 9 | 0 | 0.004 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|1 | 10 | +0.008 | +0.039 | 7 | 3 | 0.770 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|2 | 10 | -0.070 | -0.038 | 2 | 8 | 0.020 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|short|5 | 10 | -0.113 | -0.073 | 0 | 10 | 0.002 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|1 | 10 | +0.001 | +0.022 | 5 | 5 | 0.748 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|2 | 10 | +0.030 | +0.063 | 6 | 3 | 0.359 |
| caps_chain|S L-A3 offline vs S MS-TCN reference|long|5 | 10 | -0.014 | -0.033 | 4 | 6 | 0.625 |
| caps_chain|S L-A3 offline vs M L-A3 offline (EXP-21, long head)|long|1 | 10 | +0.051 | -0.004 | 4 | 6 | 0.922 |
| caps_chain|S L-A3 offline vs M L-A3 offline (EXP-21, long head)|long|2 | 10 | +0.060 | +0.059 | 5 | 4 | 0.359 |
| caps_chain|S L-A3 offline vs M L-A3 offline (EXP-21, long head)|long|5 | 10 | +0.098 | +0.126 | 7 | 3 | 0.131 |
| caps_chain|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|short|1 | 10 | +0.237 | +0.249 | 10 | 0 | 0.002 |
| caps_chain|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|short|2 | 10 | +0.043 | +0.012 | 5 | 5 | 0.770 |
| caps_chain|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|short|5 | 10 | -0.022 | -0.011 | 4 | 6 | 0.557 |
| caps_chain|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|long|1 | 10 | +0.031 | +0.069 | 6 | 4 | 0.922 |
| caps_chain|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|long|2 | 10 | +0.112 | +0.078 | 8 | 2 | 0.027 |
| caps_chain|S L-A3 offline vs M short MS-TCN (EXP-21 p3)|long|5 | 10 | +0.115 | +0.088 | 9 | 0 | 0.004 |
| phases F1@50 P0|S L-A3 offline vs S MS-TCN reference | 10 | +31.471 | +29.990 | 10 | 0 | 0.002 |
| phases F1@50 P1|S L-A3 offline vs S MS-TCN reference | 10 | +7.727 | +8.065 | 8 | 1 | 0.008 |

## Prediction check: long-group mAP@+-2 s, mean of 2 seeds, S L-A3 offline vs reference + 0.03
| decoder | S L-A3 seeds | S L-A3 mean | reference seeds | reference mean | threshold | difference |
|---|---|---|---|---|---|---|
| nms | 0.419, 0.357 | 0.388 | 0.429, 0.297 | 0.363 | 0.393 | +0.025 |
| caps_chain | 0.437, 0.350 | 0.393 | 0.433, 0.299 | 0.366 | 0.396 | +0.028 |

## Online (causal L-A3 S; selected on inner val: a3_causal_s0, plain; clip delay 2 s, decoder delay 2 s)
| group | mAP@1 / @2 / @5 [95% CI] |
|---|---|
| short | 0.275 [0.20, 0.36] / 0.494 [0.43, 0.63] / 0.742 [0.61, 0.87] |
| long | 0.292 [0.21, 0.44] / 0.364 [0.26, 0.54] / 0.587 [0.51, 0.71] |

Latency of the true positives (s): +-1: mean 4.13 std 0.55 n 158; +-2: mean 4.11 std 0.81 n 195; +-5: mean 3.88 std 1.62 n 229

Phases online (a3_causal_ens): F1@50 P0 44.7, H 59.2; acc 1 Hz P0 0.816, H 0.820

Per seed (no selection), online group mAP@1 / @2 / @5:
| run | decoder | short | long |
|---|---|---|---|
| a3_causal_s0 | plain | 0.275 / 0.494 / 0.742 | 0.292 / 0.364 / 0.587 |
| a3_causal_s0 | caps_chain | 0.254 / 0.415 / 0.566 | 0.139 / 0.188 / 0.325 |
| a3_causal_s1 | plain | 0.238 / 0.526 / 0.680 | 0.169 / 0.286 / 0.424 |
| a3_causal_s1 | caps_chain | 0.203 / 0.443 / 0.564 | 0.092 / 0.173 / 0.280 |

EXP-21 online (arm M encoder short + causal M long model), same evaluation: short 0.307 / 0.406 / 0.440; long 0.123 / 0.248 / 0.352
