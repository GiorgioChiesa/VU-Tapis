#!/usr/bin/env python3
"""
Experiment 2, arm 2.3 (true phases as input): is the gain phase *context* or phase *boundaries*?

Every Orsi phase START coincides with an event stamp, so a true phase sequence in input tells the
model *when* the phase-start events happen. For every event class: fraction of its onsets within
``--tol`` s of a phase change (over the 50 videos), and the per-class AP@5 gain of arm 2.3 over 2.1
(from ``outputs/orsi/EXP2/summary.json``). Aggregate output only.

Example:
    python tools/analyze_oracle_phase.py
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from train_hybrid import boundary_fraction, load_orsi

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-dir", default=str(ROOT / "outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all"))
    ap.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--summary", default=str(ROOT / "outputs/orsi/EXP2/summary.json"))
    ap.add_argument("--arm", default="2.3_oracle_phase")
    ap.add_argument("--ref", default="2.1_H*")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "outputs/orsi/EXP2/analysis"))
    args = ap.parse_args()
    out = Path(args.out)
    ns = SimpleNamespace(feat_dir=args.feat_dir, frame_dir=args.frame_dir, cv_fold=None, target="point", k=3,
                         sigma=2.0, streams="dynamic")
    data, _, info = load_orsi(ns, out)
    names = info["names"]
    near, tot = {}, {}
    for d in data.values():
        ph = np.asarray(d["phase"])
        for c in {c for c, _ in d["events"]}:
            on = [t for cc, t in d["events"] if cc == c]
            near[names[c]] = near.get(names[c], 0) + boundary_fraction(ph, on, args.tol) * len(on)
            tot[names[c]] = tot.get(names[c], 0) + len(on)
    fam = json.loads(Path(args.summary).read_text())["families"]
    a, r = fam[args.arm]["_per_class"], fam[args.ref]["_per_class"]
    rows = sorted(({"event": n, "boundary_frac": near[n] / tot[n], "ap_ref": r[n], "ap_arm": a[n],
                    "delta": a[n] - r[n]} for n in a if n in r and n in tot), key=lambda x: -x["boundary_frac"])
    f = np.array([x["boundary_frac"] for x in rows])
    dd = np.array([x["delta"] for x in rows])
    res = {"arm": args.arm, "ref": args.ref, "tol_s": args.tol, "per_class": rows,
           "boundary_events": {"n": int((f >= 0.5).sum()), "mean_delta": float(dd[f >= 0.5].mean())},
           "other_events": {"n": int((f < 0.5).sum()), "mean_delta": float(dd[f < 0.5].mean())},
           "spearman": {"rho": float(spearmanr(f, dd)[0]), "p": float(spearmanr(f, dd)[1])}}
    (out / "oracle_phase_boundary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))


if __name__ == "__main__":
    main()
