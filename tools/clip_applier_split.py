#!/usr/bin/env python3
"""
EXP-21 phase 3: AP of the three ``clip_`` events split by whether the TAPIS teacher detects a Clip Applier
near the event.

Teacher signal per second: max over the regions of ``m2f_score * p_TAPIS(Clip Applier)`` (0 when the second has
no region), as ``orsi_signals.py``. An event (ground truth or detection) at time t is "with Clip Applier" when
the signal over [t - W, t + W] s reaches ``--thr`` (defaults W = 2 s, thr = 0.3). Ground truth and detections
are split independently by the flag at their own time and AP is computed in each part.
Detections: ``final_offline.json`` (held-out videos, short decoder chosen on the inner val). Aggregates only.

    python tools/clip_applier_split.py --dets results/p3/fusion/final_offline.json --out results/p3/clip_applier/split.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F, ev, se  # noqa: E402

ACT_DIR = Path("/home/gchie/workspace/nas_private/data/Orsi/action_labels")
CLIP_CLASSES = ["Clip_or_coagulate_vas_deferens_-_right", "Identification_and_clipping_of_SV_arteries_-_right",
                "Metal_clip_on_right_pedicle"]


def clip_applier_signal(doc, n_seconds=None):
    """Per-second teacher signal of the Clip Applier from an ``RARPxx_actions.json`` dict.

    >>> d = {"seconds": [{"second": 1, "regions": [{"m2f_score": 0.5, "tapis_instruments": {"Clip Applier": 0.8}}]},
    ...                  {"second": 2, "regions": []}]}
    >>> clip_applier_signal(d, 4).round(2).tolist()
    [0.0, 0.4, 0.0, 0.0]
    """
    rows = doc["seconds"]
    n = n_seconds or (max(r["second"] for r in rows) + 1)
    sig = np.zeros(n)
    for r in rows:
        if 0 <= r["second"] < n and r["regions"]:
            sig[r["second"]] = max(g["m2f_score"] * g["tapis_instruments"]["Clip Applier"] for g in r["regions"])
    return sig


def flag_at(sig, t, w=2.0, thr=0.3):
    """True when the signal over [t - w, t + w] seconds reaches ``thr``.

    >>> flag_at(np.array([0, 0, 0.5, 0, 0, 0]), 4.0, 2.0, 0.3), flag_at(np.array([0, 0, 0.5, 0, 0, 0]), 5.0, 2.0, 0.3)
    (True, False)
    """
    lo, hi = max(0, int(np.floor(t - w))), min(len(sig) - 1, int(np.ceil(t + w)))
    return bool(hi >= lo and sig[lo : hi + 1].max() >= thr)


def split_ap(dets, gts, flags, classes, deltas=P.DELTAS):
    """AP per class and part. ``flags[v](t) -> bool``. Returns {class: {part: {delta: AP, 'n_gt', 'n_det'}}}."""
    out = {}
    for c in classes:
        out[c] = {}
        for part, want in (("with", True), ("without", False), ("all", None)):
            r = {"n_gt": 0, "n_det": 0}
            sc, tp = {d: [] for d in deltas}, {d: [] for d in deltas}
            for v in gts:
                keep = (lambda t, v=v: True) if want is None else (lambda t, v=v: flags[v](t) == want)
                g = [t for k, t in gts[v] if k == c and keep(t)]
                p = [(t, s) for k, t, s in dets.get(v, []) if k == c and keep(t)]
                r["n_gt"] += len(g)
                r["n_det"] += len(p)
                for d in deltas:
                    m, _, _ = se.match_events([t for t, _ in p], [s for _, s in p], g, d)
                    sc[d] += [s for _, s in p]
                    tp[d] += [float(x) for x in m]
            for d in deltas:
                r[str(d)] = ev.fast_ap(np.array(sc[d]), np.array(tp[d]), r["n_gt"])
            out[c][part] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", default=str(P.OUT / "fusion/final_offline.json"))
    ap.add_argument("--out", default=str(P.OUT / "clip_applier/split.json"))
    ap.add_argument("--set", default="held_out")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--videos", type=int, default=0)
    a = ap.parse_args()
    ctx = P.Ctx(a.set)
    if a.videos:
        ctx.videos = ctx.videos[: a.videos]
    doc = json.loads(Path(a.dets).read_text())
    dets = {v: [tuple(x) for x in doc[a.set].get(v, [])] for v in ctx.videos}
    names = F.EVENT_CLASSES
    gts = ctx.gts(names)
    flags, cover = {}, {}
    for v in ctx.videos:
        sig = clip_applier_signal(json.loads((ACT_DIR / f"{v}_actions.json").read_text()), int(ctx.labs[v].n / 4) + 1)
        flags[v] = lambda t, sig=sig: flag_at(sig, t, a.window, a.thr)
        cover[v] = float((sig >= a.thr).mean())
    classes = [names.index(c) for c in CLIP_CLASSES]
    res = split_ap(dets, gts, flags, classes)
    out = {"per_class": {names[c]: res[c] for c in classes}, "window_s": a.window, "thr": a.thr,
           "videos": len(ctx.videos), "mean_seconds_with_clip_applier": float(np.mean(list(cover.values()))),
           "mean_over_clip_classes": {part: {str(d): float(np.nanmean([res[c][part][str(d)] for c in classes]))
                                             for d in P.DELTAS} for part in ("with", "without", "all")}}
    P.dump_json(out, a.out)
    for c in classes:
        print(names[c], {p: (res[c][p]["n_gt"], round(res[c][p]["2"], 3)) for p in ("with", "without", "all")})


if __name__ == "__main__":
    main()
