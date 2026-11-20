#!/usr/bin/env python3
"""
Timeline error viewer for the 8 target events (EXP-22 L-A3 on the S latents vs the EXP-19 S MS-TCN reference,
plus any extra experiment given with --extra NAME=RUNDIR_PATTERN, where {fold} is replaced by the fold index,
and merged detection files given with --dets NAME=JSON_PATTERN, e.g. the EXP-24 hybrid).

Per held-out video of each fold: a bar of the true phases, the true events (one symbol per class) and, per model,
the predicted events with the same symbols. Predictions are matched one-to-one per class (greedy by score, within
7 s) and coloured by offset: green <= 1 s, dark green <= 5 s, orange <= 7 s, red = false positive. True events
without a match are red (false negatives); matched ones take the colour of their match.

Operating point (display only, not a metric): per model, video set and class, the score threshold with the best F1
at +-2 s on the shown videos (p3lib.best_thresholds), the same rule for every model. Decoder: nms.
Writes one self-contained HTML page (inline SVG + JS); times and labels only, no frames.

    python tools/export_timeline_viewer.py --out ~/workspace/viewer/timelines_focus8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F  # noqa: E402

EXP22 = P.REPO / "results/exp22"
NAMES = F.EVENT_CLASSES
MAXOFF = 7.0


def runs_for(fold, extra):
    R = EXP22 / f"fold{fold}"
    out = {"EXP-22 L-A3 (S features)": R / "a3_s0" / "held_out", "EXP-19 S MS-TCN (reference)": R / "reference_s0"}
    for name, pat in extra:
        d = Path(pat.format(fold=fold))
        if d.exists():
            out[name] = d
    return out


def match(preds, gts):
    """preds [(c, t, s)], gts [(c, t)] -> pred colours, gt colours (offset class per item, 'fp'/'fn')."""
    def tier(off):
        return "g1" if off <= 1 else ("g5" if off <= 5 else "o7")
    pc, gc = ["fp"] * len(preds), ["fn"] * len(gts)
    for c in set(k for k, _, _ in preds) | set(k for k, _ in gts):
        pi = sorted([i for i, p in enumerate(preds) if p[0] == c], key=lambda i: -preds[i][2])
        gi = [i for i, g in enumerate(gts) if g[0] == c]
        used = set()
        for i in pi:
            best, bo = None, MAXOFF + 1e-9
            for j in gi:
                if j in used:
                    continue
                o = abs(preds[i][1] - gts[j][1])
                if o <= bo:
                    best, bo = j, o
            if best is not None:
                used.add(best)
                pc[i] = gc[best] = tier(bo)
    return pc, gc


def segments(phase_rows, fps):
    segs, start = [], 0
    for r in range(1, len(phase_rows) + 1):
        if r == len(phase_rows) or phase_rows[r] != phase_rows[start]:
            segs.append([round(start / fps, 2), round(r / fps, 2), int(F.PHASE_IDS[int(phase_rows[start])])])
            start = r
    return segs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / "workspace/viewer/timelines_focus8"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--extra", action="append", default=[], help="NAME=DIR_PATTERN with {fold}")
    ap.add_argument("--dets", action="append", default=[], help="NAME=JSON_PATTERN with {fold}: merged detections "
                    "{video: [[class, time, score]]} (EXP-24 hybrid: results/exp24/fold{fold}/hybrid_dets_s0.json)")
    a = ap.parse_args()
    extra = [tuple(e.split("=", 1)) for e in a.extra]
    dets_extra = [tuple(e.split("=", 1)) for e in a.dets]
    data = {"events": NAMES[1:], "short": F.SHORT, "folds": {}}
    counts = {}
    for fold in [int(x) for x in a.folds.split(",")]:
        P.SPLITS = EXP22 / f"fold{fold}" / "splits.json"
        ct = P.Ctx("held_out")
        gts = ct.gts(NAMES)
        fd = {"videos": {}}
        sources = {m: ("dir", d) for m, d in runs_for(fold, extra).items()}
        sources.update({m: ("dets", Path(pat.format(fold=fold))) for m, pat in dets_extra})
        for model, (kind, d) in sources.items():
            if kind == "dir":
                dets = ct.detect(P.load_dir(d, ct.videos), NAMES, "nms")
            else:
                dets = {v: [(int(c), float(t), float(s)) for c, t, s in x] for v, x in json.loads(d.read_text()).items()}
            th = P.best_thresholds(dets, gts, range(1, len(NAMES)), 2.0)
            dets = P.apply_thresholds(dets, th)
            for v in ct.videos:
                pc, gc = match(dets[v], gts[v])
                vd = fd["videos"].setdefault(v, {"models": {}})
                vd["models"][model] = {"pred": [[int(c) - 1, round(t, 2), round(float(s), 3), col]
                                                for (c, t, s), col in zip(dets[v], pc)],
                                       "gt_col": gc}
                for col in pc + [g for g in gc if g == "fn"]:
                    counts.setdefault(model, {}).setdefault(col, 0)
                    counts[model][col] += 1
        for v in ct.videos:
            lab = ct.labs[v]
            fd["videos"][v].update(duration=round(lab.n / lab.fps, 2), phases=segments(lab.phase, lab.fps),
                                   gt=[[int(c) - 1, round(t, 2)] for c, t in gts[v]])
        data["folds"][str(fold)] = fd
        print("fold", fold, "done", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    html = (Path(__file__).with_name("timeline_viewer_template.html")).read_text()
    (out / "index.html").write_text(html.replace("/*DATA*/null", json.dumps(data)))
    (out / "counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
