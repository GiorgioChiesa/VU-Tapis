#!/usr/bin/env python3
"""
Experiment 3: how precise are the phase boundaries of a temporal model?

For every run of every arm (``<root>/<arm>/seed*/fold*/val/<patient>.npz`` of train_hybrid.py),
against the ground-truth per-second phases:
  * boundary P / R / F1 at +/- 2, 5, 10, 30 s of the argmax phase sequence (raw and after a
    ``--smooth`` s moving average of the posteriors), one-to-one matching (segment_eval.boundary_f1);
  * edit score and segmental F1@10/25/50 of the argmax phases;
  * phase-change spotting AP at 1..30 s (NMS +/- 5 s) of two soft scores: the posterior change
    0.5 * |p(t + 2) - p(t - 2)|_1 (every arm) and `probs_boundary` (arms with --boundary-head);
  * share of the onsets of the boundary event classes (``oracle_phase_boundary.json``) with an
    argmax phase change within +/- 5 s.
Mean and std over the runs of each arm. Aggregate output only.

Example:
    python tools/eval_phase_boundaries.py --arms EXP2/F_base EXP3/B1_head EXP3/B2_add
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.ndimage import uniform_filter1d

from train_hybrid import load_orsi
from tapis.evaluate import segment_eval
from tapis.evaluate import spotting_eval as se

ROOT = Path(__file__).resolve().parents[1] / "outputs/orsi"
TOLS = (2, 5, 10, 30)
DELTAS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30)


def change_score(probs, w=2):
    """0.5 * L1 distance between the phase posteriors ``w`` s after and ``w`` s before each second.

    >>> change_score(np.eye(2)[[0, 0, 0, 1, 1, 1]], w=1).tolist()
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    """
    t = np.arange(len(probs))
    after, before = probs[np.minimum(t + w, len(probs) - 1)], probs[np.maximum(t - w, 0)]
    return 0.5 * np.abs(after - before).sum(1)


def change_ap(scores, gts):
    """Spotting AP of the phase changes, ``{delta: AP}``, for per-video soft scores ``[T]``."""
    dets = {v: se.decode_events(s[:, None], 5.0, 0.0, background_classes=()) for v, s in scores.items()}
    gt = {v: [(0, float(t)) for t in segment_eval.changes(g)] for v, g in gts.items()}
    return {d: float(m) for d, m in se.spotting_map(dets, gt, DELTAS, 1, ignore_classes=())["map"].items()}


def run_metrics(val, data, bnd_ids, smooth):
    npz = {z.stem: np.load(z) for z in sorted(val.glob("RARP*.npz"))}
    gts = {p: np.asarray(data[p]["phase"])[: len(z["probs_phases"])] for p, z in npz.items()}
    m = {}
    for tag, w in (("raw", 0), (f"smooth{smooth}", smooth)):
        lab = {p: (uniform_filter1d(z["probs_phases"], w, axis=0) if w else z["probs_phases"]).argmax(1)
               for p, z in npz.items()}
        for tol in TOLS:
            m.update({f"{tag}/{k}@{tol}s": v for k, v in segment_eval.boundary_f1(lab, gts, tol).items()})
        m[f"{tag}/pred_per_gt"] = sum(len(segment_eval.changes(v)) for v in lab.values()) / max(
            1, sum(len(segment_eval.changes(v)) for v in gts.values()))
        if tag == "raw":
            m.update({f"seg/{k}": v for k, v in segment_eval.segmental_metrics(lab, gts).items()})
            near = [np.abs(segment_eval.changes(lab[p]) - t).min() <= 5 if len(segment_eval.changes(lab[p])) else False
                    for p in npz for c, t in data[p]["events"] if c in bnd_ids]
            m["boundary_events_near_pred_change@5s"] = float(np.mean(near))
    m.update({f"ap_change_score@{d}s": v for d, v in change_ap(
        {p: change_score(z["probs_phases"]) for p, z in npz.items()}, gts).items()})
    if all("probs_boundary" in z.files for z in npz.values()):
        m.update({f"ap_boundary_head@{d}s": v for d, v in change_ap(
            {p: z["probs_boundary"][:, 0] for p, z in npz.items()}, gts).items()})
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["EXP2/F_base"])
    ap.add_argument("--feat-dir", default=str(ROOT / "LONG/control/totale/untrimmed/grasp_LONG/all"))
    ap.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--boundary", default=str(ROOT / "EXP2/analysis/oracle_phase_boundary.json"))
    ap.add_argument("--smooth", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "EXP3/analysis/phase_boundaries.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ns = SimpleNamespace(feat_dir=args.feat_dir, frame_dir=args.frame_dir, cv_fold=None, target="point", k=3,
                         sigma=2.0, streams="dynamic")
    data, _, info = load_orsi(ns, out.parent)
    bnd = [r["event"] for r in json.loads(Path(args.boundary).read_text())["per_class"] if r["boundary_frac"] >= 0.5]
    bnd_ids = {info["names"].index(n) for n in bnd}
    res = {}
    for arm in args.arms:
        runs = [run_metrics(v, data, bnd_ids, args.smooth) for v in sorted((ROOT / arm).glob("seed*/fold*/val"))]
        if not runs:
            print(f"{arm}: no runs")
            continue
        res[arm] = {"n_runs": len(runs), **{k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                                            for k in runs[0]}}
        r = res[arm]
        print(f"{arm} ({len(runs)} runs): boundary F1@5 raw {r['raw/F1@5s'][0]:.3f} smooth {r[f'smooth{args.smooth}/F1@5s'][0]:.3f}"
              f" | R@5 {r['raw/R@5s'][0]:.3f} P@5 {r['raw/P@5s'][0]:.3f} | edit {r['seg/edit'][0]:.1f}"
              f" | AP change@5 {r['ap_change_score@5s'][0]:.3f}"
              + (f" head {r['ap_boundary_head@5s'][0]:.3f}" if "ap_boundary_head@5s" in r else ""), flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
