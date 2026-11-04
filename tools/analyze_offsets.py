#!/usr/bin/env python3
"""
Experiment 4: how far are the detections from the true onsets, and what does the displacement do?

For every run of every arm (``<root>/<arm>/seed*/fold*/val/<patient>.npz`` of train_hybrid.py):
  * spotting mAP at 1, 2, 3, 5 s of the raw event scores (``probs_steps_raw``, or ``probs_steps`` for
    arms without --offset-head) and of the displaced ones (``probs_steps``);
  * localisation error of the best-scored same-class detection within +/- 5 s of every onset
    (share <= 1 s, <= 2 s, 3-5 s, mean signed error), for boundary event classes and the others;
  * ceilings on the raw scores: displacement by the true per-class offset within +/- 3 and 5 s
    (``displace_scores`` with ``offset_targets``), i.e. a perfect offset head.
Mean over the runs of each arm. Aggregate output only.

Example:
    python tools/analyze_offsets.py --arms EXP2/F_base EXP4/O1_agnostic3 EXP4/K5
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from eval_untrimmed import idle_index
from train_hybrid import displace_scores, load_orsi, offset_targets
from tapis.evaluate import spotting_eval as se

ROOT = Path(__file__).resolve().parents[1] / "outputs/orsi"
DELTAS = (1, 2, 3, 5)


def errors(dets, gts, tol=5):
    """Signed error (s) of the best-scored same-class detection within ``tol`` s of each onset.

    >>> errors([(1, 4, 0.9), (1, 7, 0.95), (2, 3, 0.5)], [(1, 5.0), (2, 9.0)])
    [2]
    """
    out = []
    for c, t in gts:
        cand = [(s, tt) for cc, tt, s in dets if cc == c and abs(tt - t) <= tol]
        if cand:
            out.append(int(round(max(cand)[1] - t)))
    return out


def run_metrics(val, data, names, bnd):
    bg = tuple(idle_index(names))
    fg = [c for c in range(len(names)) if c not in bg]
    npz = {z.stem: np.load(z) for z in sorted(val.glob("RARP*.npz"))}
    gts = {p: data[p]["events"] for p in npz}
    scores = {"raw": {p: (z["probs_steps_raw"] if "probs_steps_raw" in z.files else z["probs_steps"]) for p, z in npz.items()},
              "displaced": {p: z["probs_steps"] for p, z in npz.items()}}
    for r in (3, 5):
        scores[f"ceiling_R{r}"] = {p: displace_scores(s, offset_targets(len(s), gts[p], len(names), r, True)[0], r, fg, bg)
                                   for p, s in scores["raw"].items()}
    m = {}
    for k, sc in scores.items():
        dets = {p: se.decode_events(s, 5.0, 0.0, background_classes=bg) for p, s in sc.items()}
        res = se.spotting_map(dets, gts, DELTAS, len(names), ignore_classes=bg)["map"]
        m.update({f"{k}/mAP@{d}": float(res[d]) for d in DELTAS})
        if k in ("raw", "displaced"):
            for grp, keep in (("boundary", lambda c: c in bnd), ("other", lambda c: c not in bnd)):
                e = np.array([x for p in sc for x in errors(dets[p], [g for g in gts[p] if keep(g[0])])])
                m[f"{k}/{grp}/err<=1"] = float(np.mean(np.abs(e) <= 1))
                m[f"{k}/{grp}/err<=2"] = float(np.mean(np.abs(e) <= 2))
                m[f"{k}/{grp}/err3-5"] = float(np.mean(np.abs(e) >= 3))
                m[f"{k}/{grp}/bias"] = float(e.mean())
                m[f"{k}/{grp}/hist"] = np.bincount(e + 5, minlength=11).tolist()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["EXP2/F_base"])
    ap.add_argument("--feat-dir", default=str(ROOT / "LONG/control/totale/untrimmed/grasp_LONG/all"))
    ap.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--boundary", default=str(ROOT / "EXP2/analysis/oracle_phase_boundary.json"))
    ap.add_argument("--out", default=str(ROOT / "EXP4/analysis/offsets.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ns = SimpleNamespace(feat_dir=args.feat_dir, frame_dir=args.frame_dir, cv_fold=None, target="point", k=3,
                         sigma=2.0, streams="dynamic")
    data, _, info = load_orsi(ns, out.parent)
    names = info["names"]
    bnd = {names.index(r["event"]) for r in json.loads(Path(args.boundary).read_text())["per_class"]
           if r["boundary_frac"] >= 0.5}
    res = {}
    for arm in args.arms:
        runs = [run_metrics(v, data, names, bnd) for v in sorted((ROOT / arm).glob("seed*/fold*/val"))]
        if not runs:
            print(f"{arm}: no runs")
            continue
        res[arm] = {"n_runs": len(runs)}
        for k in runs[0]:
            v = [r[k] for r in runs]
            res[arm][k] = np.sum(v, 0).tolist() if k.endswith("hist") else [float(np.mean(v)), float(np.std(v))]
        r = res[arm]
        print(f"{arm} ({len(runs)}): raw mAP@1/2/5 {r['raw/mAP@1'][0]:.3f}/{r['raw/mAP@2'][0]:.3f}/{r['raw/mAP@5'][0]:.3f}"
              f" | displaced {r['displaced/mAP@1'][0]:.3f}/{r['displaced/mAP@2'][0]:.3f}/{r['displaced/mAP@5'][0]:.3f}"
              f" | ceiling R3 @1 {r['ceiling_R3/mAP@1'][0]:.3f} R5 @1 {r['ceiling_R5/mAP@1'][0]:.3f}"
              f" | other err<=1 raw {r['raw/other/err<=1'][0]:.2f} disp {r['displaced/other/err<=1'][0]:.2f}", flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
