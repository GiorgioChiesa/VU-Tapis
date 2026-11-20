# EXP-25: fully online hybrid vs offline hybrid vs offline reference, 5 folds x 10 held-out videos (50 videos)

AP in [0, 1]; 95% CI = per-video bootstrap within fold (1000, same resamples for every model), 5-fold mean CI = bootstrap of the mean of the 5 folds. Online = decoder delay 2 s + clip delay 2 s. Offline hybrid / reference = EXP-24 (reference = mean of its 2 seeds, nms).

## Choices made on the inner val
| fold | short part (cand / decoder) | inner short score | long part (cand / decoder) | inner long score | phases from | short nms cand |
|---|---|---|---|---|---|---|
| 0 | ens / plain | 0.496 | ens / plain | 0.241 | ens | ens |
| 1 | s0 / plain | 0.488 | s1 / plain | 0.334 | ens | ens |
| 2 | ens / plain | 0.520 | s1 / plain | 0.501 | ens | ens |
| 3 | s1 / plain | 0.502 | s0 / plain | 0.512 | ens | ens |
| 4 | s0 / plain | 0.537 | s1 / plain | 0.400 | ens | s0 |

## 8-event mean (mean of the 8 class APs) (AP@1 / @2 / @5 s) per fold and 5-fold mean
| fold | model | @1 | @2 | @5 |
|---|---|---|---|---|
| 0 | online hybrid | 0.186 [0.15, 0.27] | 0.423 [0.35, 0.53] | 0.697 [0.65, 0.75] |
| 0 | offline hybrid | 0.354 [0.31, 0.45] | 0.562 [0.49, 0.67] | 0.700 [0.61, 0.81] |
| 0 | offline reference | 0.373 [0.31, 0.48] | 0.564 [0.49, 0.69] | 0.735 [0.65, 0.84] |
| 1 | online hybrid | 0.175 [0.12, 0.27] | 0.328 [0.25, 0.46] | 0.651 [0.58, 0.74] |
| 1 | offline hybrid | 0.380 [0.32, 0.49] | 0.610 [0.53, 0.73] | 0.720 [0.65, 0.82] |
| 1 | offline reference | 0.374 [0.33, 0.45] | 0.583 [0.53, 0.66] | 0.708 [0.65, 0.78] |
| 2 | online hybrid | 0.088 [0.06, 0.17] | 0.287 [0.22, 0.42] | 0.522 [0.44, 0.67] |
| 2 | offline hybrid | 0.432 [0.32, 0.61] | 0.645 [0.52, 0.80] | 0.740 [0.64, 0.86] |
| 2 | offline reference | 0.378 [0.29, 0.51] | 0.562 [0.48, 0.68] | 0.693 [0.61, 0.80] |
| 3 | online hybrid | 0.226 [0.19, 0.33] | 0.436 [0.36, 0.55] | 0.648 [0.58, 0.74] |
| 3 | offline hybrid | 0.344 [0.27, 0.48] | 0.612 [0.54, 0.73] | 0.736 [0.66, 0.84] |
| 3 | offline reference | 0.344 [0.28, 0.46] | 0.565 [0.49, 0.66] | 0.714 [0.64, 0.81] |
| 4 | online hybrid | 0.130 [0.08, 0.24] | 0.353 [0.30, 0.47] | 0.645 [0.58, 0.75] |
| 4 | offline hybrid | 0.299 [0.24, 0.43] | 0.517 [0.45, 0.64] | 0.750 [0.71, 0.83] |
| 4 | offline reference | 0.291 [0.25, 0.39] | 0.533 [0.48, 0.62] | 0.723 [0.68, 0.79] |
| 5-fold mean | online hybrid | 0.161 [0.14, 0.23] | 0.365 [0.35, 0.44] | 0.632 [0.61, 0.70] |
| 5-fold mean | offline hybrid | 0.362 [0.34, 0.43] | 0.589 [0.58, 0.65] | 0.729 [0.70, 0.79] |
| 5-fold mean | offline reference | 0.352 [0.33, 0.41] | 0.562 [0.54, 0.62] | 0.715 [0.69, 0.76] |
| 5-fold mean | online hybrid - offline hybrid | -0.201 [-0.24, -0.16] | -0.224 [-0.24, -0.19] | -0.097 [-0.12, -0.07] |
| 5-fold mean | online hybrid - offline reference | -0.191 [-0.24, -0.14] | -0.196 [-0.23, -0.15] | -0.082 [-0.11, -0.05] |

## short group mAP (AP@1 / @2 / @5 s) per fold and 5-fold mean
| fold | model | @1 | @2 | @5 |
|---|---|---|---|---|
| 0 | online hybrid | 0.099 [0.06, 0.19] | 0.454 [0.37, 0.57] | 0.773 [0.70, 0.85] |
| 0 | offline hybrid | 0.531 [0.44, 0.65] | 0.766 [0.65, 0.87] | 0.869 [0.77, 0.94] |
| 0 | offline reference | 0.531 [0.44, 0.65] | 0.766 [0.65, 0.87] | 0.869 [0.77, 0.94] |
| 0 | causal short MS-TCN alone (online decoding) | 0.099 [0.06, 0.19] | 0.454 [0.37, 0.57] | 0.773 [0.70, 0.85] |
| 1 | online hybrid | 0.154 [0.09, 0.25] | 0.372 [0.26, 0.55] | 0.771 [0.69, 0.87] |
| 1 | offline hybrid | 0.500 [0.42, 0.61] | 0.772 [0.68, 0.87] | 0.879 [0.81, 0.95] |
| 1 | offline reference | 0.500 [0.42, 0.61] | 0.772 [0.68, 0.87] | 0.879 [0.81, 0.95] |
| 1 | causal short MS-TCN alone (online decoding) | 0.154 [0.09, 0.25] | 0.372 [0.26, 0.55] | 0.771 [0.69, 0.87] |
| 2 | online hybrid | 0.100 [0.08, 0.16] | 0.334 [0.28, 0.46] | 0.656 [0.53, 0.83] |
| 2 | offline hybrid | 0.521 [0.42, 0.67] | 0.770 [0.66, 0.90] | 0.855 [0.76, 0.94] |
| 2 | offline reference | 0.521 [0.42, 0.67] | 0.770 [0.66, 0.90] | 0.855 [0.76, 0.94] |
| 2 | causal short MS-TCN alone (online decoding) | 0.100 [0.08, 0.16] | 0.334 [0.28, 0.46] | 0.656 [0.53, 0.83] |
| 3 | online hybrid | 0.240 [0.16, 0.35] | 0.535 [0.42, 0.66] | 0.800 [0.71, 0.89] |
| 3 | offline hybrid | 0.504 [0.38, 0.66] | 0.833 [0.75, 0.92] | 0.892 [0.81, 0.96] |
| 3 | offline reference | 0.504 [0.38, 0.66] | 0.833 [0.75, 0.92] | 0.892 [0.81, 0.96] |
| 3 | causal short MS-TCN alone (online decoding) | 0.240 [0.16, 0.35] | 0.535 [0.42, 0.66] | 0.800 [0.71, 0.89] |
| 4 | online hybrid | 0.088 [0.07, 0.18] | 0.389 [0.31, 0.55] | 0.742 [0.65, 0.87] |
| 4 | offline hybrid | 0.483 [0.40, 0.60] | 0.763 [0.68, 0.86] | 0.917 [0.86, 0.97] |
| 4 | offline reference | 0.483 [0.40, 0.60] | 0.763 [0.68, 0.86] | 0.917 [0.86, 0.97] |
| 4 | causal short MS-TCN alone (online decoding) | 0.088 [0.07, 0.18] | 0.389 [0.31, 0.55] | 0.742 [0.65, 0.87] |
| 5-fold mean | online hybrid | 0.136 [0.11, 0.20] | 0.417 [0.38, 0.50] | 0.748 [0.71, 0.81] |
| 5-fold mean | offline hybrid | 0.508 [0.47, 0.58] | 0.781 [0.73, 0.84] | 0.882 [0.84, 0.92] |
| 5-fold mean | offline reference | 0.508 [0.47, 0.58] | 0.781 [0.73, 0.84] | 0.882 [0.84, 0.92] |
| 5-fold mean | causal short MS-TCN alone (online decoding) | 0.136 [0.11, 0.20] | 0.417 [0.38, 0.50] | 0.748 [0.71, 0.81] |
| 5-fold mean | causal short MS-TCN alone (nms decoding, offline decoder) | 0.129 [0.10, 0.20] | 0.436 [0.39, 0.51] | 0.763 [0.72, 0.83] |
| 5-fold mean | online hybrid - offline hybrid | -0.371 [-0.42, -0.32] | -0.364 [-0.39, -0.30] | -0.134 [-0.15, -0.10] |
| 5-fold mean | online hybrid - offline reference | -0.371 [-0.42, -0.32] | -0.364 [-0.39, -0.30] | -0.134 [-0.15, -0.10] |
| 5-fold mean | causal short MS-TCN alone (online decoding) - offline reference | -0.371 [-0.42, -0.32] | -0.364 [-0.39, -0.30] | -0.134 [-0.15, -0.10] |

## long group mAP (AP@1 / @2 / @5 s) per fold and 5-fold mean
| fold | model | @1 | @2 | @5 |
|---|---|---|---|---|
| 0 | online hybrid | 0.272 [0.22, 0.37] | 0.391 [0.28, 0.57] | 0.620 [0.53, 0.72] |
| 0 | offline hybrid | 0.178 [0.10, 0.34] | 0.357 [0.29, 0.51] | 0.532 [0.43, 0.71] |
| 0 | offline reference | 0.215 [0.15, 0.38] | 0.363 [0.27, 0.57] | 0.601 [0.49, 0.78] |
| 1 | online hybrid | 0.196 [0.11, 0.36] | 0.284 [0.17, 0.51] | 0.531 [0.42, 0.65] |
| 1 | offline hybrid | 0.259 [0.14, 0.47] | 0.448 [0.33, 0.64] | 0.562 [0.47, 0.70] |
| 1 | offline reference | 0.247 [0.19, 0.35] | 0.394 [0.33, 0.49] | 0.537 [0.46, 0.65] |
| 2 | online hybrid | 0.076 [0.03, 0.19] | 0.239 [0.13, 0.43] | 0.388 [0.23, 0.63] |
| 2 | offline hybrid | 0.343 [0.17, 0.61] | 0.519 [0.33, 0.76] | 0.625 [0.44, 0.83] |
| 2 | offline reference | 0.234 [0.13, 0.42] | 0.354 [0.24, 0.53] | 0.531 [0.41, 0.70] |
| 3 | online hybrid | 0.211 [0.18, 0.33] | 0.337 [0.27, 0.49] | 0.495 [0.41, 0.64] |
| 3 | offline hybrid | 0.185 [0.13, 0.35] | 0.390 [0.28, 0.62] | 0.579 [0.45, 0.78] |
| 3 | offline reference | 0.184 [0.11, 0.32] | 0.297 [0.18, 0.47] | 0.536 [0.43, 0.70] |
| 4 | online hybrid | 0.172 [0.08, 0.36] | 0.318 [0.21, 0.50] | 0.547 [0.44, 0.72] |
| 4 | offline hybrid | 0.115 [0.06, 0.32] | 0.271 [0.16, 0.49] | 0.583 [0.52, 0.71] |
| 4 | offline reference | 0.100 [0.05, 0.22] | 0.303 [0.23, 0.45] | 0.528 [0.48, 0.62] |
| 5-fold mean | online hybrid | 0.186 [0.17, 0.26] | 0.314 [0.28, 0.41] | 0.516 [0.47, 0.61] |
| 5-fold mean | offline hybrid | 0.216 [0.17, 0.33] | 0.397 [0.37, 0.50] | 0.576 [0.52, 0.69] |
| 5-fold mean | offline reference | 0.196 [0.17, 0.29] | 0.342 [0.32, 0.43] | 0.547 [0.51, 0.63] |
| 5-fold mean | online hybrid - offline hybrid | -0.030 [-0.10, 0.04] | -0.083 [-0.15, -0.03] | -0.060 [-0.09, -0.02] |
| 5-fold mean | online hybrid - offline reference | -0.011 [-0.07, 0.05] | -0.028 [-0.07, 0.02] | -0.030 [-0.09, 0.02] |

## Summary, mAP@2: 8-event mean and short group, per fold and 5-fold mean
| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|---|
| online hybrid | 0.423 | 0.328 | 0.287 | 0.436 | 0.353 | 0.365 |
| offline hybrid | 0.562 | 0.610 | 0.645 | 0.612 | 0.517 | 0.589 |
| offline reference | 0.564 | 0.583 | 0.562 | 0.565 | 0.533 | 0.562 |
| short group: causal short alone (online) | 0.454 | 0.372 | 0.334 | 0.535 | 0.389 | 0.417 |
| short group: offline reference | 0.766 | 0.772 | 0.770 | 0.833 | 0.763 | 0.781 |

## AP per class: per-fold AP@2 (folds 0-4) and 5-fold mean AP@1 / @2 / @5 with CI
| class | model | AP@2 folds 0 / 1 / 2 / 3 / 4 | @1 | @2 | @5 |
|---|---|---|---|---|---|
| Grasping_catheter_tip | online hybrid | 0.652 / 0.579 / 0.167 / 0.651 / 0.127 | 0.239 [0.11, 0.45] | 0.435 [0.31, 0.61] | 0.720 [0.62, 0.86] |
| Grasping_catheter_tip | offline hybrid | 0.555 / 0.660 / 0.638 / 0.745 / 0.544 | 0.357 [0.23, 0.55] | 0.628 [0.47, 0.78] | 0.851 [0.74, 0.96] |
| Grasping_catheter_tip | offline reference | 0.555 / 0.660 / 0.638 / 0.745 / 0.544 | 0.357 [0.23, 0.55] | 0.628 [0.47, 0.78] | 0.851 [0.74, 0.96] |
| Grasping_catheter_tip | causal short MS-TCN alone (online decoding) | 0.652 / 0.579 / 0.167 / 0.651 / 0.127 | 0.239 [0.11, 0.45] | 0.435 [0.31, 0.61] | 0.720 [0.62, 0.86] |
| Clip_or_coagulate_vas_deferens_-_right | online hybrid | 0.488 / 0.192 / 0.248 / 0.621 / 0.545 | 0.068 [0.03, 0.15] | 0.419 [0.35, 0.52] | 0.685 [0.58, 0.81] |
| Clip_or_coagulate_vas_deferens_-_right | offline hybrid | 0.775 / 0.771 / 0.731 / 0.911 / 0.837 | 0.562 [0.45, 0.69] | 0.805 [0.76, 0.87] | 0.851 [0.80, 0.91] |
| Clip_or_coagulate_vas_deferens_-_right | offline reference | 0.775 / 0.771 / 0.731 / 0.911 / 0.837 | 0.562 [0.45, 0.69] | 0.805 [0.76, 0.87] | 0.851 [0.80, 0.91] |
| Clip_or_coagulate_vas_deferens_-_right | causal short MS-TCN alone (online decoding) | 0.488 / 0.192 / 0.248 / 0.621 / 0.545 | 0.068 [0.03, 0.15] | 0.419 [0.35, 0.52] | 0.685 [0.58, 0.81] |
| Identification_and_clipping_of_SV_arteries_-_right | online hybrid | 0.389 / 0.409 / 0.542 / 0.516 / 0.455 | 0.137 [0.13, 0.18] | 0.462 [0.43, 0.53] | 0.780 [0.73, 0.84] |
| Identification_and_clipping_of_SV_arteries_-_right | offline hybrid | 0.827 / 0.874 / 0.848 / 0.871 / 0.804 | 0.540 [0.50, 0.60] | 0.845 [0.80, 0.88] | 0.902 [0.86, 0.94] |
| Identification_and_clipping_of_SV_arteries_-_right | offline reference | 0.827 / 0.874 / 0.848 / 0.871 / 0.804 | 0.540 [0.50, 0.60] | 0.845 [0.80, 0.88] | 0.902 [0.86, 0.94] |
| Identification_and_clipping_of_SV_arteries_-_right | causal short MS-TCN alone (online decoding) | 0.389 / 0.409 / 0.542 / 0.516 / 0.455 | 0.137 [0.13, 0.18] | 0.462 [0.43, 0.53] | 0.780 [0.73, 0.84] |
| Metal_clip_on_right_pedicle | online hybrid | 0.286 / 0.308 / 0.380 / 0.353 / 0.429 | 0.101 [0.08, 0.14] | 0.351 [0.32, 0.42] | 0.808 [0.79, 0.84] |
| Metal_clip_on_right_pedicle | offline hybrid | 0.907 / 0.782 / 0.864 / 0.807 / 0.867 | 0.573 [0.53, 0.62] | 0.845 [0.83, 0.87] | 0.925 [0.91, 0.94] |
| Metal_clip_on_right_pedicle | offline reference | 0.907 / 0.782 / 0.864 / 0.807 / 0.867 | 0.573 [0.53, 0.62] | 0.845 [0.83, 0.87] | 0.925 [0.91, 0.94] |
| Metal_clip_on_right_pedicle | causal short MS-TCN alone (online decoding) | 0.286 / 0.308 / 0.380 / 0.353 / 0.429 | 0.101 [0.08, 0.14] | 0.351 [0.32, 0.42] | 0.808 [0.79, 0.84] |
| Start_dissection | online hybrid | 0.404 / 0.222 / 0.219 / 0.226 / 0.321 | 0.182 [0.13, 0.29] | 0.279 [0.23, 0.39] | 0.527 [0.47, 0.62] |
| Start_dissection | offline hybrid | 0.399 / 0.351 / 0.749 / 0.359 / 0.236 | 0.260 [0.15, 0.45] | 0.419 [0.31, 0.60] | 0.606 [0.52, 0.72] |
| Start_dissection | offline reference | 0.389 / 0.377 / 0.481 / 0.187 / 0.288 | 0.200 [0.12, 0.35] | 0.344 [0.24, 0.53] | 0.643 [0.58, 0.73] |
| Incision_of_Denonvilliers_fascia | online hybrid | 0.406 / 0.306 / 0.426 / 0.642 / 0.250 | 0.206 [0.12, 0.38] | 0.406 [0.30, 0.58] | 0.714 [0.58, 0.86] |
| Incision_of_Denonvilliers_fascia | offline hybrid | 0.370 / 0.635 / 0.474 / 0.444 / 0.528 | 0.241 [0.17, 0.44] | 0.490 [0.41, 0.65] | 0.735 [0.65, 0.87] |
| Incision_of_Denonvilliers_fascia | offline reference | 0.537 / 0.647 / 0.445 / 0.426 / 0.671 | 0.277 [0.20, 0.46] | 0.545 [0.45, 0.69] | 0.745 [0.67, 0.84] |
| Transection_of_the_urethra | online hybrid | 0.610 / 0.324 / 0.013 / 0.338 / 0.450 | 0.203 [0.11, 0.37] | 0.347 [0.18, 0.57] | 0.494 [0.35, 0.68] |
| Transection_of_the_urethra | offline hybrid | 0.435 / 0.544 / 0.293 / 0.314 / 0.074 | 0.160 [0.11, 0.35] | 0.332 [0.24, 0.54] | 0.522 [0.42, 0.73] |
| Transection_of_the_urethra | offline reference | 0.280 / 0.492 / 0.221 / 0.184 / 0.147 | 0.170 [0.10, 0.30] | 0.265 [0.19, 0.41] | 0.515 [0.43, 0.66] |
| Stitch_in_bladder | online hybrid | 0.146 / 0.283 / 0.298 / 0.143 / 0.249 | 0.151 [0.09, 0.30] | 0.224 [0.14, 0.41] | 0.331 [0.24, 0.52] |
| Stitch_in_bladder | offline hybrid | 0.226 / 0.262 / 0.561 / 0.444 / 0.246 | 0.203 [0.14, 0.36] | 0.348 [0.26, 0.51] | 0.440 [0.35, 0.59] |
| Stitch_in_bladder | offline reference | 0.245 / 0.061 / 0.270 / 0.392 / 0.105 | 0.137 [0.08, 0.24] | 0.215 [0.16, 0.31] | 0.284 [0.25, 0.39] |

## Latency of the true positives (s) = emission time + clip delay (2 s) - ground-truth time, pooled over the 5 folds
| part | tolerance | mean | std | n TP | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|---|---|---|
| hybrid | +-1 | 4.06 | 0.59 | 541 | 4.00 | 4.15 | 4.10 | 4.01 | 4.01 |
| short | +-1 | 4.03 | 0.59 | 460 | 3.96 | 4.13 | 4.09 | 4.00 | 3.98 |
| long | +-1 | 4.17 | 0.60 | 81 | 4.11 | 4.28 | 4.16 | 4.10 | 4.25 |
| hybrid | +-2 | 4.41 | 1.12 | 1007 | 4.46 | 4.38 | 4.55 | 4.36 | 4.32 |
| short | +-2 | 4.42 | 1.13 | 886 | 4.54 | 4.39 | 4.55 | 4.39 | 4.29 |
| long | +-2 | 4.32 | 1.00 | 121 | 4.12 | 4.36 | 4.57 | 4.12 | 4.51 |
| hybrid | +-5 | 4.62 | 1.84 | 1473 | 4.83 | 4.68 | 4.54 | 4.62 | 4.49 |
| short | +-5 | 4.70 | 1.78 | 1301 | 5.00 | 4.72 | 4.56 | 4.78 | 4.51 |
| long | +-5 | 4.04 | 2.14 | 172 | 3.97 | 4.32 | 4.39 | 3.31 | 4.28 |

## Pooled paired Wilcoxon over the 50 videos (per-video group mAP; mean diff, better/worse, p)
| comparison | group | tol | n | mean diff | better/worse | p |
|---|---|---|---|---|---|---|
| online hybrid - offline hybrid | mean8 | @1 | 50 | -0.207 | 6/44 | 0.0000 |
| online hybrid - offline hybrid | mean8 | @2 | 50 | -0.201 | 7/43 | 0.0000 |
| online hybrid - offline hybrid | mean8 | @5 | 50 | -0.069 | 12/38 | 0.0000 |
| online hybrid - offline hybrid | short | @1 | 50 | -0.362 | 1/49 | 0.0000 |
| online hybrid - offline hybrid | short | @2 | 50 | -0.303 | 2/48 | 0.0000 |
| online hybrid - offline hybrid | short | @5 | 50 | -0.090 | 6/44 | 0.0000 |
| online hybrid - offline hybrid | long | @1 | 50 | -0.060 | 22/23 | 0.1631 |
| online hybrid - offline hybrid | long | @2 | 50 | -0.103 | 16/33 | 0.0066 |
| online hybrid - offline hybrid | long | @5 | 50 | -0.052 | 20/30 | 0.0411 |
| online hybrid - offline reference | mean8 | @1 | 50 | -0.185 | 8/42 | 0.0000 |
| online hybrid - offline reference | mean8 | @2 | 50 | -0.167 | 4/46 | 0.0000 |
| online hybrid - offline reference | mean8 | @5 | 50 | -0.062 | 16/34 | 0.0000 |
| online hybrid - offline reference | short | @1 | 50 | -0.362 | 1/49 | 0.0000 |
| online hybrid - offline reference | short | @2 | 50 | -0.303 | 2/48 | 0.0000 |
| online hybrid - offline reference | short | @5 | 50 | -0.090 | 6/44 | 0.0000 |
| online hybrid - offline reference | long | @1 | 50 | -0.015 | 19/29 | 0.6666 |
| online hybrid - offline reference | long | @2 | 50 | -0.034 | 21/28 | 0.3103 |
| online hybrid - offline reference | long | @5 | 50 | -0.039 | 24/26 | 0.0968 |
| causal short MS-TCN alone (online decoding) - offline reference | short | @2 | 50 | -0.303 | 2/48 | 0.0000 |

## Phases (causal L-A3 online vs offline), F1@50 (mean over videos) and 1 Hz accuracy
| fold | online P0 (raw) | online H (causal hysteresis) | offline hybrid P0 | offline hybrid P1 | offline ref P0 | offline ref P1 | online acc P0 | online acc H |
|---|---|---|---|---|---|---|---|---|
| 0 | 44.7 | 59.2 | 73.6 | 80.2 | 40.8 | 70.9 | 0.816 | 0.820 |
| 1 | 47.5 | 64.4 | 72.0 | 77.4 | 41.4 | 74.5 | 0.850 | 0.839 |
| 2 | 44.1 | 59.7 | 66.7 | 76.0 | 45.6 | 72.9 | 0.826 | 0.818 |
| 3 | 44.3 | 63.5 | 70.4 | 78.6 | 40.6 | 74.3 | 0.860 | 0.858 |
| 4 | 50.9 | 64.0 | 74.1 | 83.5 | 47.6 | 74.3 | 0.850 | 0.852 |
| mean | 46.3 | 62.2 | 71.4 | 79.1 | 43.2 | 73.4 | 0.840 | 0.837 |

| phases comparison (F1@50, 50 videos) | n | mean diff | better/worse | p |
|---|---|---|---|---|
| online H - offline hybrid P1 | 50 | -17.0 | 2/47 | 0.0000 |
| online H - offline reference P1 | 50 | -11.2 | 5/45 | 0.0000 |
| online P0 - offline hybrid P0 | 50 | -25.0 | 2/48 | 0.0000 |

## Prediction check (pre-registered in EXPERIMENT.md, EXP-25)
| quantity | value |
|---|---|
| online_hybrid_mean8_at2 | 0.3654 |
| offline_hybrid_mean8_at2 | 0.5890 |
| threshold_0.85x_offline | 0.5007 |
| ratio | 0.6203 |
| holds_mean8 | False |
| latency_mean_at2 | 4.4101 |
| holds_latency | True |
| causal_short_at2 | 0.4169 |
| reference_short_at2 | 0.7809 |
| secondary_diff | -0.3640 |
| holds_secondary | False |
