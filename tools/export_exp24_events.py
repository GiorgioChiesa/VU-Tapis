#!/usr/bin/env python3
"""
Per-event viewer of the EXP-24 hybrid (reference S MS-TCN short events + L-A3 long events) on the held-out videos of all 5 folds.
Same layout as tools/export_exp21_viewer.py (page split by true event class; per event a 16-frame grid at 4 Hz and a probability
strip), plus an MP4 clip of the same 16 frames to the right of the grid and strip plots at 200 dpi (displayed width unchanged).

Detections: results/exp24/fold<i>/hybrid_dets_s0.json (reference seed 0 short + the L-A3 candidate/decoder chosen on the inner val).
Per-class score threshold = best F1 at +-2 s on the held-out detections of the OTHER 4 folds (the reference has no inner-val
predictions, so the EXP-21 inner-val rule is not available; the test fold never chooses its own threshold).
Curve: reference_s0 probability for the short classes, the chosen L-A3 candidate (s0/s1/ens) for the long classes.
Items per class and fold: 3 TP, 3 missed, 2 false alarms (top by score) -> up to 15/15/10 per class over 5 folds, as EXP-21.
Clips: H.264 yuv420p +faststart, 320x180 (frames_4fps are 16:9 squeezed to 224x224; the grid also uses 160x90), 4 fps = real time.
Patient frames and clips are written only under --out on ORSI-GPU2.

    python tools/export_exp24_events.py --out ~/workspace/viewer/EXP24_hybrid/events
"""
import argparse
import html
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp22_report as R22  # noqa: E402
from exp24_hybrid import l3_runs  # noqa: E402
import p3lib as P  # noqa: E402
from p3lib import F  # noqa: E402

FRAMES4 = Path("/home/gchie/workspace/data/orsi_tensors_4fps")
FFMPEG = "/home/gchie/miniforge3/envs/Vu-Tapis/lib/python3.13/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
NAMES = F.EVENT_CLASSES
SHORT_C = [c for c in range(1, len(NAMES)) if NAMES[c] in F.SHORT]
DPI, FIGSIZE = 200, (6, 1.8)       # displayed at 600 px wide (width attribute), 1200 px of pixels
CSS = ("body{font-family:sans-serif;margin:1.5em;max-width:1500px}img{max-width:100%}h2{margin-top:1.6em;border-top:2px solid #ccc;padding-top:.4em}"
       "small{color:#555}.card{display:flex;gap:14px;align-items:flex-start;border-bottom:1px solid #ddd;padding:6px 0}"
       ".card .l{flex:1;min-width:0}.card .r{flex:0 0 320px}.card video{width:320px;height:180px;background:#000}.plot{width:600px}")


def strip_image(frames, rows, size=(160, 90)):
    from PIL import Image
    ims = [np.asarray(Image.fromarray(np.ascontiguousarray(np.asarray(frames[r]).transpose(1, 2, 0)[..., ::-1])).resize(size)) for r in rows]
    return np.concatenate([np.concatenate(ims[:8], 1), np.concatenate(ims[8:], 1)], 0)


def write_mp4(path, frames, rows, size=(320, 180), fps=4, crf=28):
    """16 BGR frames [3,H,W] -> RGB, resized to ``size``, H.264 yuv420p +faststart at ``fps`` (real time for the 4 Hz source)."""
    from PIL import Image
    w, h = size
    raw = b"".join(np.asarray(Image.fromarray(np.ascontiguousarray(np.asarray(frames[r]).transpose(1, 2, 0)[..., ::-1])).resize(size, Image.LANCZOS)).tobytes()
                   for r in rows)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)]
    subprocess.run(cmd, input=raw, check=True)


def trace_png(path, second, prob, center_s, gts, dets, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.abs(second - center_s) <= 15
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(second[m], prob[m], lw=1)
    for t in gts:
        if abs(t - center_s) <= 15:
            ax.axvline(t, color="green", lw=1.5, label="ground truth")
    for t, s in dets:
        if abs(t - center_s) <= 15:
            ax.axvline(t, color="red", ls="--", lw=1, label=f"detection {s:.2f}")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def event_items(dets, gts, thr, k_tp=3, k_miss=3, k_fa=2):
    items = {}
    for c in range(1, len(NAMES)):
        lst = []
        for v in gts:
            p = [(t, s) for k, t, s in dets.get(v, []) if k == c and s >= thr[c]]
            g = [t for k, t in gts[v] if k == c]
            pairs = P.match_with_gt([t for t, _ in p], [s for _, s in p], g, 2.0)
            mp, mg = {i for i, _ in pairs}, {j for _, j in pairs}
            lst += [("TP", v, g[j], p[i][1]) for i, j in pairs]
            lst += [("missed", v, g[j], None) for j in range(len(g)) if j not in mg]
            lst += [("false alarm", v, p[i][0], p[i][1]) for i in range(len(p)) if i not in mp]
        tp = [x for x in lst if x[0] == "TP"][:k_tp]
        ms = [x for x in lst if x[0] == "missed"][:k_miss]
        fa = sorted([x for x in lst if x[0] == "false alarm"], key=lambda x: -x[3])[:k_fa]
        items[c] = tp + ms + fa
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/home/gchie/workspace/viewer/EXP24_hybrid/events")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    out = Path(a.out)
    folds = [int(x) for x in a.folds.split(",")]
    R = P.REPO / "results/exp24"
    dets, gts, ctxs, runs = {}, {}, {}, {}
    for f in folds:
        P.SPLITS = R22.EXP22 / f"fold{f}/splits.json"
        ct = P.Ctx("held_out")
        ctxs[f], gts[f] = ct, ct.gts(NAMES)
        dets[f] = {v: [tuple(x) for x in d] for v, d in json.loads((R / f"fold{f}/hybrid_dets_s0.json").read_text()).items()}
        cand = json.loads((R / f"fold{f}/fold.json").read_text())["selection"]["chosen"].split("|")[0]
        ref = P.load_dir(R22.EXP22 / f"fold{f}/reference_s0", ct.videos)
        runs[f] = (ref, l3_runs(R22.EXP22 / f"fold{f}", ct)[cand], cand)
    thr = {}
    for f in folds:
        oth = [g for g in folds if g != f]
        d = {f"{g}/{v}": x for g in oth for v, x in dets[g].items()}
        gt = {f"{g}/{v}": x for g in oth for v, x in gts[g].items()}
        thr[f] = P.best_thresholds(d, gt, list(range(1, len(NAMES))), 2.0)
    cards, ncount, nclip = {c: [] for c in range(1, len(NAMES))}, {c: {} for c in range(1, len(NAMES))}, 0
    cache = {}
    for f in folds:
        items = event_items(dets[f], gts[f], thr[f])
        ref, l3, cand = runs[f]
        for c, its in items.items():
            name = NAMES[c]
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            for kind, v, t, s in its:
                stem = f"{kind.replace(' ', '_')}_f{f}_{v}_t{int(round(t * 4))}"
                if v not in cache:
                    cache.clear()
                    cache[v] = np.load(FRAMES4 / v / "frames_4fps.npy", mmap_mode="r")
                fr = cache[v]
                rows = F.clip_rows(int(round(t * 4)), len(fr))
                from PIL import Image
                Image.fromarray(strip_image(fr, rows)).save(d / f"{stem}.jpg", quality=80)
                write_mp4(d / f"{stem}.mp4", fr, rows)
                nclip += 1
                src = ref[v] if c in SHORT_C else l3[v]
                gt_t = [x for kk, x in gts[f][v] if kk == c]
                trace_png(d / f"{stem}.png", src["second"], src["probs_steps"][:, c], t, gt_t,
                          [(x, sc) for kk, x, sc in dets[f][v] if kk == c and sc >= thr[f][c]],
                          f"{name[:40]} / {kind} / fold {f} / {'reference S MS-TCN' if c in SHORT_C else 'L-A3 ' + cand}")
                ph = int(ctxs[f].labs[v].phase[min(ctxs[f].labs[v].n - 1, int(round(t * 4)))])
                cards[c].append(
                    f'<div class="card"><div class="l"><b>{kind}</b> {v} (fold {f}) t={t:.1f}s phase P{F.PHASE_IDS[ph]}'
                    + (f" score {s:.2f}" if s is not None else "") + f'<br><img src="{stem}.jpg"><br><img class="plot" src="{stem}.png"></div>'
                    f'<div class="r"><video controls muted loop preload="none" width="320" height="180"><source src="{stem}.mp4" type="video/mp4"></video></div></div>')
                ncount[c][kind] = ncount[c].get(kind, 0) + 1
    index = []
    for c, cs in cards.items():
        name = NAMES[c]
        ths = ", ".join(f"f{f} {thr[f][c]:.2f}" for f in folds)
        (out / name / "index.html").write_text(
            f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body><p><a href='../index.html'>all classes</a></p>"
            f"<h2>{html.escape(name)}</h2><p><small>EXP-24 hybrid, held-out videos of the 5 folds; threshold per fold (best F1 at +-2 s on the other 4 folds): {ths}; "
            f"16 frames at 4 Hz (4 s) centred on the event or detection, clip = the same frames at 4 fps; curve = probability of the "
            f"{'reference S MS-TCN' if c in SHORT_C else 'L-A3 (chosen candidate)'}</small></p>{''.join(cs)}</body></html>")
        index.append(f'<li><a href="{html.escape(name)}/index.html">{html.escape(name)}</a> {ncount[c]}</li>')
    (out / "index.html").write_text(
        f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body><p><a href='../index.html'>EXP-24 hybrid</a></p>"
        f"<h2>EXP-24 hybrid: per-event strips and clips (5 folds, 50 held-out videos)</h2><p><small>generated {time.strftime('%F %T')}; "
        f"patient frames stay on ORSI-GPU2</small></p><ul>{''.join(index)}</ul></body></html>")
    print("events", sum(len(v) for v in cards.values()), "clips", nclip, "->", out / "index.html")


if __name__ == "__main__":
    main()
