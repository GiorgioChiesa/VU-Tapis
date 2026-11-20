#!/usr/bin/env python3
"""
EXP-21 viewer: ``<out>/index.html`` (tables of the reference, arms M and M0, long models, fusion, exploratory
short MS-TCN, Clip Applier split, online evaluation; offline and online confusion matrices; correlation figures)
and per-event 4 Hz strips of the held-out videos (adapted from tools/export_error_viewer.py, which assumes 1 fps).
It renders whatever exists when it runs: a missing input becomes a "pending" note, so the final regeneration job
can run after any subset of the other jobs. Patient frames are written only under ``--out`` on ORSI-GPU2
(served by the http.server of port 8765); nothing leaves the machine.

    python tools/export_exp21_viewer.py --out /home/gchie/workspace/viewer/EXP21_focus8 [--no-frames]
Strips: ``events/<class>/index.html`` + ``<item>.jpg`` (16 frames, 4 Hz, centred on the event / detection, 2 rows of
8) and ``<item>.png`` (class probability of the encoder head, +-15 s, with the ground truth and detections).
"""

import argparse
import html
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F  # noqa: E402

FOCUS8 = P.FOCUS8
P3 = P.OUT
FRAMES4 = Path("/home/gchie/workspace/data/orsi_tensors_4fps")
CSS = ("body{font-family:sans-serif;margin:1.5em;max-width:1300px}table{border-collapse:collapse;margin:.6em 0;font-size:13px}"
       "td,th{border:1px solid #bbb;padding:2px 6px;text-align:right}th{background:#eee}td:first-child,th:first-child{text-align:left}"
       "img{max-width:100%}.pending{color:#a60}h2{margin-top:1.6em;border-top:2px solid #ccc;padding-top:.4em}"
       ".grid img{max-width:48%}small{color:#555}")


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def f3(x, ci=None):
    if x is None:
        return "-"
    s = f"{x:.3f}"
    return s + (f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else "")


def table(header, rows):
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def spot_rows(label, rep, variants=("nms", "caps", "caps_chain", "phase_caps_chain")):
    """Rows ``[label, decoder, short @1 @2 @5, long @1 @2 @5]`` of an eval_focus8 report (or one arm result)."""
    out = []
    for var in variants:
        r = rep.get(var) if rep is not None and var in rep else None
        if r is None:
            continue
        out.append([label, var] + group_cells(r))
    return out


def group_cells(r):
    cells = []
    for g in ("short", "long"):
        gm, ci = r["group_map"].get(g), (r.get("group_map_ci95") or {}).get(g)
        for d in P.DELTAS:
            cells.append(f3(gm[str(d)], ci[str(d)] if ci else None) if gm else "-")
    return cells


def per_class_table(rows):
    """rows: {label: spotting result}; AP at +-1/2/5 per target class."""
    header = ["predictions"] + [f"{n[:22]} @{d}" for n in F.TARGETS for d in P.DELTAS]
    body = []
    for lab, r in rows.items():
        cells = []
        for n in F.TARGETS:
            for d in P.DELTAS:
                v = r["per_class_ap"].get(str(d), {}).get(n)
                cells.append(f3(v))
        body.append([lab] + cells)
    return table(header, body)


def pending(what):
    return f'<p class="pending">pending: {html.escape(what)}</p>'


def section_reference():
    ref = sorted((FOCUS8 / "results/reference/v2026-11-19").glob("*_focus8/eval_test.json"))
    if not ref:
        return pending("reference evaluation (results/reference/v2026-11-19/*_focus8/eval_test.json)")
    rows, pc = [], {}
    for f in ref:
        rep = load(f)
        name = f.parent.name.replace("_focus8", "")
        rows += spot_rows(name, rep, ("nms", "caps_chain"))
        pc[name + " nms"] = rep["nms"]
    return (table(["predictions", "decoder", "short @1", "short @2", "short @5", "long @1", "long @2", "long @5"], rows)
            + per_class_table(pc))


def section_arms():
    rows, pc, ph = [], {}, []
    for arm, d in (("M enc", FOCUS8 / "results/M/fold0/untrimmed/eval"), ("M0 enc", FOCUS8 / "results/M0/fold0/untrimmed/eval")):
        for st in ("held_out", "inner_val"):
            rep = load(d / f"{st}.json")
            if rep:
                rows += spot_rows(f"{arm} ({st})", rep, ("nms", "caps", "caps_chain"))
                if st == "held_out":
                    pc[arm + " nms"], pc[arm + " caps"] = rep["nms"], rep["caps"]
                    p = rep.get("phases", {})
                    ph.append([arm, f3(p.get("row_accuracy")), f3(p.get("row_macro_f1")),
                               f"{p.get('F1@50_mean_video', {}).get('P0', float('nan')):.1f}",
                               f"{p.get('F1@50_mean_video', {}).get('P1', float('nan')):.1f}"])
    if not rows:
        return pending("arm M / M0 encoder-head evaluation")
    return (table(["predictions", "decoder", "short @1", "short @2", "short @5", "long @1", "long @2", "long @5"], rows)
            + per_class_table(pc) + table(["predictions", "row acc", "row macro F1", "F1@50 P0", "F1@50 P1"], ph))


def section_long():
    rows, ph = [], []
    for d in sorted((FOCUS8 / "results/M/fold0/long").glob("*_s[01]")):
        rep = load(d / "eval/held_out.json")
        if rep:
            rows += spot_rows(d.name, rep, ("nms", "caps_chain"))
            p = rep.get("phases", {})
            ph.append([d.name, f3(p.get("row_accuracy")), f"{p.get('F1@50_mean_video', {}).get('P0', float('nan')):.1f}",
                       f"{p.get('F1@50_mean_video', {}).get('P1', float('nan')):.1f}"])
    if not rows:
        return pending("long models (results/M/fold0/long/*/eval/held_out.json)")
    return (table(["run (test)", "decoder", "short @1", "short @2", "short @5", "long @1", "long @2", "long @5"], rows)
            + table(["run (test)", "row acc", "F1@50 P0", "F1@50 P1"], ph))


def wil(t):
    return f"p={t['p']:.3f} (n={t['n_videos']}, mean diff {t['mean_diff']:+.3f}, better/worse {t['n_better']}/{t['n_worse']})"


def section_fusion():
    e, p = load(P3 / "fusion/events.json"), load(P3 / "fusion/phases.json")
    if not e:
        return pending("fusion (results/p3/fusion/events.json)")
    sel = e["selection"]
    out = ["<p>Selected on the inner val: " + html.escape(json.dumps({k: v for k, v in sel.items()
                                                                       if k != "b_candidates_inner_val"})) + "</p>"]
    names = {"a": "(a) encoder", "b": "(b) long model", "c": "(c) confidence-weighted", "cp": "(c') global weight"}
    rows = [[names[k], e["selection"].get(k, {}).get("decoder", "-")] + group_cells(e["test"][k])[3:] for k in names]
    out.append("<h3>Long events, test (10 videos)</h3>" + table(["arm", "decoder", "long @1", "long @2", "long @5"], rows))
    out.append(per_class_table({names[k]: e["test"][k] for k in names}))
    tests = [[k] + [wil(e["tests"][k][str(d)]) for d in P.DELTAS] for k in e["tests"]]
    out.append("<h3>Paired Wilcoxon over videos (per-video long AP)</h3>" + table(["test", "@1", "@2", "@5"], tests))
    s = e.get("short")
    if s:
        out.append("<h3>Short events, encoder heads, test</h3>" + table(
            ["decoder (inner val)", "short @1", "short @2", "short @5"], [[s["decoder"]] + group_cells(s["test"])[:3]]))
    if p:
        rows = [[names[k], f3(p["test"][k]["row_accuracy"]), f3(p["test"][k]["acc_1hz_mean_video"]["P0"]),
                 f3(p["test"][k]["acc_1hz_mean_video"]["P1"]), f"{p['test'][k]['F1@50_mean_video']['P0']:.1f}",
                 f"{p['test'][k]['F1@50_mean_video']['P1']:.1f}"] for k in names]
        out.append("<h3>Phases, test</h3>" + table(["arm", "row acc", "acc 1 Hz P0", "acc 1 Hz P1", "F1@50 without P1",
                                                    "F1@50 with P1"], rows))
        tests = [[k] + [wil(p["tests"][k][m]) for m in ("P0", "P1")] for k in p["tests"]]
        out.append(table(["test (per-video F1@50)", "without P1", "with P1"], tests))
        out.append("<small>phase selection: " + html.escape(json.dumps(p["selection"])) + "</small>")
    return "".join(out)


def section_short_mstcn():
    rows, pc = [], {}
    for d in sorted((P3 / "short_mstcn").glob("s[01]")):
        rep = load(d / "eval/held_out.json")
        if rep:
            rows += spot_rows(f"short MS-TCN {d.name}", rep, ("nms", "caps", "caps_chain"))
            pc[f"short MS-TCN {d.name} nms"] = rep["nms"]
    if not rows:
        return pending("exploratory short-event MS-TCN (results/p3/short_mstcn/s*/eval/held_out.json)")
    return ("<p>Exploratory, not pre-registered: MS-TCN dilate3 recipe of the reference on the M 4 Hz features.</p>"
            + table(["predictions", "decoder", "short @1", "short @2", "short @5", "long @1", "long @2", "long @5"], rows)
            + per_class_table(pc))


def section_clip():
    c = load(P3 / "clip_applier/split.json")
    if not c:
        return pending("Clip Applier split (results/p3/clip_applier/split.json)")
    rows = []
    for n, r in c["per_class"].items():
        for part in ("with", "without", "all"):
            x = r[part]
            rows.append([n[:40], part, x["n_gt"], x["n_det"]] + [f3(x[str(d)]) for d in P.DELTAS])
    m = c["mean_over_clip_classes"]
    rows += [["mean of the 3 clip classes", part, "", ""] + [f3(m[part][str(d)]) for d in P.DELTAS] for part in m]
    return (f"<p>Clip Applier = teacher signal >= {c['thr']} within +-{c['window_s']} s; detections of the selected "
            f"offline short decoder, {c['videos']} test videos.</p>"
            + table(["class", "part", "n GT", "n det", "AP @1", "AP @2", "AP @5"], rows))


def section_online():
    o = load(P3 / "online/online.json")
    if not o:
        return pending("online evaluation (results/p3/online/online.json)")
    t = o["test"]
    rows = [[g] + [f3(t["events"]["group_map"][g][str(d)], (t["events"].get("group_map_ci95") or {}).get(g, {}).get(str(d)))
                   for d in P.DELTAS] for g in t["events"]["group_map"]]
    lat = [[d, f"{v['mean']:.2f}", f"{v['std']:.2f}", v["n_tp"]] for d, v in t["latency"].items()]
    ph = t["phases"]
    return ("<p>Declared: " + html.escape(json.dumps(o["declared"])) + "<br>Selected on the inner val: "
            + html.escape(json.dumps(o["selection"])) + "</p>"
            + table(["group", "mAP @1", "mAP @2", "mAP @5"], rows)
            + "<p>Latency of the true positives (emission time + clip delay - ground truth, s)</p>"
            + table(["delta", "mean", "std", "n TP"], lat)
            + table(["online phases", "F1@50 argmax", "F1@50 causal hysteresis", "acc 1 Hz argmax", "acc 1 Hz hysteresis"],
                    [["test", f"{ph['F1@50_mean_video']['P0']:.1f}", f"{ph['F1@50_mean_video']['H']:.1f}",
                      f3(ph['acc_1hz_mean_video']['P0']), f3(ph['acc_1hz_mean_video']['H'])]])
            + per_class_table({"online": t["events"]}))


def imgs(out, src_dir, pattern, label):
    found = sorted(Path(src_dir).glob(pattern))
    if not found:
        return pending(label)
    html_ = []
    for f in found:
        dst = out / "figures" / f.relative_to(P3).as_posix().replace("/", "_")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
        html_.append(f'<figure><img src="figures/{dst.name}"><figcaption><small>{html.escape(f.name)}</small></figcaption></figure>')
    return '<div class="grid">' + "".join(html_) + "</div>"


# ----------------------------------------------------------------------------------------- strips
def strip_image(frames, center, n, size=(160, 90)):
    """16 frames at 4 Hz centred at row ``center`` -> one RGB array of 2 rows of 8 (BGR input, uint8).

    >>> fr = np.zeros((40, 9, 16, 3), np.uint8); fr[:, :, :, 0] = 255   # blue in BGR
    >>> s = strip_image(fr, 20, 40, size=(8, 4)); s.shape, s[0, 0].tolist()
    ((8, 64, 3), [0, 0, 255])
    """
    from PIL import Image

    rows = F.clip_rows(center, n)
    ims = [np.asarray(Image.fromarray(np.asarray(frames[r])[..., ::-1].copy()).resize(size)) for r in rows]
    w, h = size
    return np.concatenate([np.concatenate(ims[:8], 1), np.concatenate(ims[8:], 1)], 0)


def trace_png(path, second, prob, center_s, gts, dets, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.abs(second - center_s) <= 15
    fig, ax = plt.subplots(figsize=(6, 1.8))
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
    fig.savefig(path, dpi=90)
    plt.close(fig)


def event_items(dets, gts, thr, k_fa=10, k_other=15):
    """Per class: TP / missed (ground truth) and false alarms (top by score) at +-2 s on the thresholded detections."""
    items = {}
    for c, name in enumerate(F.EVENT_CLASSES):
        if c == 0:
            continue
        lst = []
        for v in gts:
            p = [(t, s) for k, t, s in dets.get(v, []) if k == c and s >= thr.get(c, 0.5)]
            g = [t for k, t in gts[v] if k == c]
            pairs = P.match_with_gt([t for t, _ in p], [s for _, s in p], g, 2.0)
            mp, mg = {i for i, _ in pairs}, {j for _, j in pairs}
            lst += [("TP", v, g[j], p[i][1]) for i, j in pairs]
            lst += [("missed", v, g[j], None) for j in range(len(g)) if j not in mg]
            lst += [("false alarm", v, p[i][0], p[i][1]) for i in range(len(p)) if i not in mp]
        tp = [x for x in lst if x[0] == "TP"][:k_other]
        ms = [x for x in lst if x[0] == "missed"][:k_other]
        fa = sorted([x for x in lst if x[0] == "false alarm"], key=lambda x: -x[3])[:k_fa]
        items[name] = tp + ms + fa
    return items


def export_strips(out):
    fo = load(P3 / "fusion/final_offline.json")
    if not fo:
        return pending("per-event strips (need results/p3/fusion/final_offline.json)")
    cv, ct = P.Ctx("inner_val"), P.Ctx("held_out")
    gts_v, gts_t = cv.gts(F.EVENT_CLASSES), ct.gts(F.EVENT_CLASSES)
    dv = {v: [tuple(x) for x in fo["inner_val"].get(v, [])] for v in gts_v}
    dt = {v: [tuple(x) for x in fo["held_out"].get(v, [])] for v in gts_t}
    thr = P.best_thresholds(dv, gts_v, list(range(1, len(F.EVENT_CLASSES))), 2.0)
    items = event_items(dt, gts_t, thr)
    enc = {v: dict(np.load(P.ENC_DIR / f"{v}.npz")) for v in ct.videos}
    frames = {}
    index = []
    for ci, (name, its) in enumerate(items.items(), start=1):
        d = out / "events" / name
        d.mkdir(parents=True, exist_ok=True)
        cards = []
        for k, (kind, v, t, s) in enumerate(its):
            stem = f"{kind.replace(' ', '_')}_{v}_t{int(round(t * 4))}"
            if v not in frames:
                frames[v] = np.load(FRAMES4 / v / "frames_4fps.npy", mmap_mode="r")
            from PIL import Image

            Image.fromarray(strip_image(frames[v], int(round(t * 4)), len(frames[v]))).save(d / f"{stem}.jpg", quality=80)
            gt_t = [x for kk, x in gts_t[v] if kk == ci]
            trace_png(d / f"{stem}.png", enc[v]["second"], enc[v]["probs_steps"][:, ci], t, gt_t,
                      [(x, sc) for kk, x, sc in dt[v] if kk == ci and sc >= thr.get(ci, 0.5)], f"{name[:40]} / {kind}")
            ph = int(ct.labs[v].phase[min(ct.labs[v].n - 1, int(round(t * 4)))])
            cards.append(f'<div><b>{kind}</b> {v} t={t:.1f}s phase P{F.PHASE_IDS[ph]}'
                         + (f" score {s:.2f}" if s is not None else "")
                         + f'<br><img src="{stem}.jpg"><br><img src="{stem}.png"></div>')
        (d / "index.html").write_text(f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body><h2>{html.escape(name)}</h2>"
                                      f"<p><small>threshold {thr.get(ci, 0.5):.2f} (inner val), +-2 s; 16 frames at 4 Hz centred on the "
                                      f"event or detection; curve = encoder head probability</small></p>{''.join(cards)}</body></html>")
        n = {k: sum(1 for x in its if x[0] == k) for k in ("TP", "missed", "false alarm")}
        index.append(f'<li><a href="events/{html.escape(name)}/index.html">{html.escape(name)}</a> {n}</li>')
    return "<ul>" + "".join(index) + "</ul>"


def main():
    global P3
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/home/gchie/workspace/viewer/EXP21_focus8")
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--p3", default=str(P3), help="results root of phase 3 (smoke tests use results/p3_smoke)")
    a = ap.parse_args()
    P3 = Path(a.p3).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    secs = [("Reference: EXP-19 S fold 0 re-evaluated on the revised labels", section_reference()),
            ("Arms M and M0: encoder heads (per clip, 4 Hz)", section_arms()),
            ("Long-context models on the M latents (offline runs, causal runs)", section_long()),
            ("Fusion of encoder and long model (events, phases)", section_fusion()),
            ("Exploratory (not pre-registered): MS-TCN for the short events on the M features", section_short_mstcn()),
            ("Clip Applier split of the clip_ events", section_clip()),
            ("Online evaluation (causal long model, center clip with 2 s delay)", section_online()),
            ("Confusion matrices, offline", imgs(out, P3 / "confusion/offline", "*.png", "offline confusion matrices")),
            ("Confusion matrices, online", imgs(out, P3 / "confusion/online", "*.png", "online confusion matrices")),
            ("Cross-correlations: true labels + TAPIS teacher", imgs(out, P3 / "corr", "*_true.png", "correlations (true)")),
            ("Cross-correlations: same videos (10 test), true labels + teacher", imgs(out, P3 / "corr", "*_true_test10.png", "correlations (true, test10)")),
            ("Cross-correlations: predicted heads (10 test videos)", imgs(out, P3 / "corr", "*_pred.png", "correlations (pred)"))]
    if not a.no_frames:
        secs.append(("Per-event 4 Hz strips (test videos)", export_strips(out)))
    body = "".join(f"<h2>{html.escape(t)}</h2>{h}" for t, h in secs)
    (out / "index.html").write_text(
        f"<html><head><meta charset=utf-8><title>EXP-21 focus8</title><style>{CSS}</style></head><body>"
        f"<h1>EXP-21: multi-task fine-tune on the 8 target events, fold 0</h1>"
        f"<p><small>labels v2026-11-19_4hz, test = 10 held-out videos, bootstrap CI over videos; generated "
        f"{time.strftime('%F %T')}; patient frames stay on ORSI-GPU2</small></p>{body}</body></html>")
    pend = sum("class=\"pending\"" in h for _, h in secs)
    (out / "summary.json").write_text(json.dumps({"generated": time.strftime("%F %T"), "sections": [t for t, _ in secs],
                                                  "pending_sections": pend}, indent=1))
    print("written", out / "index.html", "pending sections:", pend)


if __name__ == "__main__":
    main()
