#!/usr/bin/env python3
"""
Aggregate statistics of the ORSI_AUG training augmentation on real Orsi clips (no image is saved
or shown): how often each operation fires, intensity / saturation before and after, mean absolute
change, the largest within-clip difference of the augmentation residual between frames that were
identical before augmentation, and the per-clip cost. Output: one json.

Example:
    python tools/clip_augment_stats.py --cfg results/L/LONG/cv_ft/fold0/config.yaml --n 200 \
        --out results/aug_stats_L.json
"""

import argparse
import contextlib
import io
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "detectron2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from tapis.config.defaults import get_cfg
from tapis.datasets import clip_augment as ca
from tapis.datasets.orsi import Orsi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="training config yaml (e.g. a cv_finetune fold config)")
    ap.add_argument("--n", type=int, default=200, help="clips sampled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("opts", nargs="*", help="KEY VALUE overrides")
    args = ap.parse_args()

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = str(Path(args.out).parent / "aug_stats_tmp")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = Orsi(cfg, "train", load=False)
        ds._load_data()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    idx = random.Random(args.seed).sample(range(len(ds)), min(args.n, len(ds)))

    fired = Counter()
    for _ in range(10000):  # operation frequencies from the sampler itself
        p = ca.sample_params(cfg.ORSI_AUG, 224, 224)
        fired.update(k for k, v in p.items() if v not in (False, 0.0, None))
    shifts = Counter(ca.temporal_shift(int(cfg.ORSI_AUG.TEMPORAL_JITTER_ROWS)) for _ in range(10000))

    raw_m, aug_m, raw_sat, aug_sat, dabs, consist, cost = [], [], [], [], [], [], []
    for i in idx:
        cfg.ORSI_AUG.ENABLE = False
        raw = ds[i][0][0]  # C,T,H,W, keyframe-centred
        x = raw.permute(1, 0, 2, 3)
        t0 = time.time()
        p = ca.sample_params(cfg.ORSI_AUG, x.shape[-2], x.shape[-1])
        y = ca.apply(x, p)
        cost.append(time.time() - t0)
        # consistency: one frame repeated must stay identical after augmentation
        rep = ca.apply(x[:1].repeat(len(x), 1, 1, 1), p)
        consist.append(float((rep - rep[:1]).abs().max()))
        raw_m.append(float(x.mean()))
        aug_m.append(float(y.mean()))
        raw_sat.append(float((x >= 0.99).float().mean()))
        aug_sat.append(float((y >= 0.99).float().mean()))
        dabs.append(float((y - x).abs().mean()))
    q = lambda v: dict(mean=float(np.mean(v)), p10=float(np.percentile(v, 10)), p50=float(np.median(v)),
                       p90=float(np.percentile(v, 90)))
    rep = {
        "cfg": args.cfg, "clips": len(idx), "dataset_rows": len(ds),
        "orsi_aug": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.ORSI_AUG.items()},
        "op_frequency": {k: v / 10000 for k, v in sorted(fired.items())},
        "temporal_shift_hist": {str(k): v / 10000 for k, v in sorted(shifts.items())},
        "intensity_raw": q(raw_m), "intensity_aug": q(aug_m),
        "saturated_frac_raw": q(raw_sat), "saturated_frac_aug": q(aug_sat),
        "mean_abs_change": q(dabs),
        "within_clip_max_diff_of_identical_frames": max(consist),
        "seconds_per_clip": q(cost),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("clips", "op_frequency", "mean_abs_change",
                                          "within_clip_max_diff_of_identical_frames", "seconds_per_clip")}, indent=1))


if __name__ == "__main__":
    main()
