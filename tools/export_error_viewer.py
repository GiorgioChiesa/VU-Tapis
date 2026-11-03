#!/usr/bin/env python3
"""
Export a browsable folder of event-level errors of a cross-validated temporal head.

For every fold of ``--temporal-root`` (``fold*/val`` = predictions on the held-out
videos, written by ``train_temporal.py --cv-fold``) the events are decoded as in
``eval_constraints.py`` (``--variant``, threshold = the fold's best F1@delta) and
matched to the ground truth within +-delta seconds:

  1. same class, greedy by score (``spotting_eval.match_events``) -> ``X--X``
  2. leftover detections take the closest leftover ground-truth event of another
     class                                                         -> ``X--Y``
  3. leftover detections near an already found event of their class
                                                                   -> ``X--X_duplicato``
     other leftover detections (false alarms)                      -> ``nessun_evento--Y``
  4. leftover ground-truth events                                  -> ``X--mancato``

Output (``--out``, outside the repository: it contains patient frames):

  README.md, index.html, confusion_matrix.png, confusion_matrix_phases.png, summary.json
  results/<true>--<pred>/index.html
  results/<true>--<pred>/<RARPxx>_t<second>_fold<i>/{00..15}.jpg, strip.jpg, info.txt

The jpgs are the clip the encoder saw for that second (same ``get_sequence`` as
training and ``infer_untrimmed.py``), centred on the detection (on the ground
truth for missed events).

Example:
    python tools/export_error_viewer.py --temporal-root outputs/orsi/TEMPORAL/cv_ft_dilate3 \
        --out /home/gchie/workspace/viewer/MSTCN_dilate3_ftenc_capchain
"""

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from eval_constraints import detections, prepare
from summarize_temporal import metrics
from tapis.datasets import utils as data_utils
from tapis.evaluate import spotting_eval as se

NO_EVENT, MISSED = "nessun_evento", "mancato"


def safe(name):
    return name.strip().replace(" ", "_").replace("/", "-")


def match_video(dets, gts, delta):
    """Event-level assignment of one video: list of dicts (see module docstring)."""
    items = []
    by_c_pred, by_c_gt = defaultdict(list), defaultdict(list)
    for c, t, s in dets:
        by_c_pred[c].append((t, s))
    for c, t in gts:
        by_c_gt[c].append(t)
    left_pred, left_gt, found = [], [], defaultdict(list)
    for c in set(by_c_pred) | set(by_c_gt):
        p = by_c_pred.get(c, [])
        g = by_c_gt.get(c, [])
        tp, off, matched = se.match_events([t for t, _ in p], [s for _, s in p], g, delta)
        k = 0
        for (t, s), ok in zip(p, tp):
            if ok:
                items.append(dict(kind="corretto", true=c, pred=c, t_gt=t - off[k], t_pred=t, score=s))
                found[c].append(t - off[k])
                k += 1
            else:
                left_pred.append((c, t, s))
        left_gt += [(c, t) for t, m in zip(g, matched) if not m]
    used = np.zeros(len(left_gt), dtype=bool)
    for c, t, s in sorted(left_pred, key=lambda d: -d[2]):
        dist = np.array([abs(tg - t) if (not u and cg != c) else np.inf for (cg, tg), u in zip(left_gt, used)])
        if dist.size and dist.min() <= delta:
            j = int(dist.argmin())
            used[j] = True
            items.append(dict(kind="scambio", true=left_gt[j][0], pred=c, t_gt=left_gt[j][1], t_pred=t, score=s))
        elif any(abs(tf - t) <= delta for tf in found[c]):
            items.append(dict(kind="duplicato", true=c, pred=c, t_gt=None, t_pred=t, score=s))
        else:
            items.append(dict(kind="falso_allarme", true=None, pred=c, t_gt=None, t_pred=t, score=s))
    for (c, t), u in zip(left_gt, used):
        if not u:
            items.append(dict(kind="mancato", true=c, pred=None, t_gt=t, t_pred=None, score=None))
    return items


def folder_name(it, names):
    true = names[it["true"]] if it["true"] is not None else NO_EVENT
    if it["kind"] == "mancato":
        return f"{safe(true)}--{MISSED}"
    pred = safe(names[it["pred"]]) + ("_duplicato" if it["kind"] == "duplicato" else "")
    return f"{safe(true)}--{pred}"


def to_jpg(t):
    return Image.fromarray((t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy())


def export_item(it, frame_dir, num_frames, sampling_rate, seq_mode):
    labels = it["labels"]
    center = it["row"]
    seq = data_utils.get_sequence(center, num_frames * sampling_rate, sampling_rate, len(labels), mode=seq_mode)
    d = it["dir"]
    d.mkdir(parents=True, exist_ok=True)
    tiles = []
    for k, idx in enumerate(seq):
        path = Path(frame_dir) / it["patient"] / "Video_1fps" / Path(labels["frame_path"].iloc[idx]).name
        img = to_jpg(torch.load(path, map_location="cpu", weights_only=True))
        img.save(d / f"{k:02d}.jpg", quality=90)
        tile = img.copy()
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, 92, 14], fill=(0, 0, 0))
        draw.text((3, 2), f"t={labels['second'].iloc[idx]} ({idx - center:+d})", fill=(255, 255, 255))
        if idx == center:
            draw.rectangle([0, 0, tile.width - 1, tile.height - 1], outline=(255, 210, 0), width=4)
        tiles.append(tile)
    side = int(np.ceil(np.sqrt(len(tiles))))
    w, h = tiles[0].size
    strip = Image.new("RGB", (side * w, side * h))
    for k, tile in enumerate(tiles):
        strip.paste(tile, ((k % side) * w, (k // side) * h))
    strip.save(d / "strip.jpg", quality=85)
    (d / "info.txt").write_text(it["info"])


def plot_matrix(mat, rows, cols, path, title, normalize=True, annotate=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = mat / np.maximum(mat.sum(1, keepdims=True), 1) if normalize else mat
    size = max(8, 0.33 * max(len(rows), len(cols)) + 3)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xlabel("predetto")
    ax.set_ylabel("reale")
    ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(len(rows)):
            for j in range(len(cols)):
                if mat[i, j]:
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=5,
                            color="white" if m[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="frazione della riga" if normalize else "conteggio")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_html(out, pairs, kinds_of, n_items, title):
    rows = "\n".join(
        f'<tr class="{kinds_of[p]}"><td><a href="results/{p}/index.html">{html.escape(p)}</a></td>'
        f"<td>{kinds_of[p]}</td><td>{n}</td></tr>"
        for p, n in pairs
    )
    (out / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;margin:16px;max-width:1200px}} td,th{{padding:2px 8px}}
tr.corretto td{{background:#e6f4ea}} tr.scambio td{{background:#fde7e7}} tr.falso_allarme td{{background:#fff4e0}}
tr.mancato td{{background:#eee}} tr.duplicato td{{background:#f3e8fd}} img{{max-width:100%}}</style></head><body>
<h1>{title}</h1><p><a href="README.md">README.md</a> · <a href="../EVENTI.md">descrizione degli eventi</a> · {n_items} inferenze ·
<a href="confusion_matrix.png">confusion matrix eventi</a> · <a href="confusion_matrix_phases.png">fasi</a></p>
<img src="confusion_matrix.png"><h2>Cartelle reale--predetto</h2>
<table><tr><th>cartella</th><th>tipo</th><th>n</th></tr>{rows}</table></body></html>""")


def write_pair_html(d, items):
    cards = "\n".join(
        f'<div class="c"><a href="{it["dir"].name}/"><img src="{it["dir"].name}/strip.jpg" loading="lazy"></a>'
        f'<pre>{html.escape(it["info"])}</pre></div>'
        for it in sorted(items, key=lambda i: (i["patient"], i["t_pred"] or i["t_gt"]))
    )
    (d / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{d.name}</title>
<style>body{{font-family:sans-serif;margin:16px}} .c{{display:inline-block;width:460px;margin:6px;vertical-align:top}}
.c img{{width:448px}} pre{{font-size:11px;white-space:pre-wrap}}</style></head><body>
<p><a href="../../index.html">indice</a></p><h1>{html.escape(d.name)} ({len(items)})</h1>
<p>Riquadro giallo = secondo dell'inferenza (detection, o evento vero se mancato); i 16 frame sono il clip dato all'encoder.</p>
{cards}</body></html>""")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temporal-root", default="outputs/orsi/TEMPORAL/cv_ft_dilate3")
    parser.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="caps_chain")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--phase-smooth", type=int, default=1)
    parser.add_argument("--gate-floor", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-frames", action="store_true", help="only matrices and counts")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = f"{args.variant}|a={args.alpha:g}|s={args.phase_smooth}|f={args.gate_floor:g}"
    fold_dirs = sorted(Path(args.temporal_root).glob("fold*"))

    items, checks, fold_metrics, class_ap = [], [], {}, defaultdict(list)
    names = ph_names = None
    ph_true, ph_pred = [], []
    for fd in fold_dirs:
        fold = fd.name.removeprefix("fold")
        prep = prepare(fd / "val", args.frame_dir)
        if names is None:
            names, ph_names = prep.names, prep.ph_names
        assert prep.names == names and prep.ph_names == ph_names, f"{fd}: class order differs"
        res = json.loads((fd / "val" / "constraints_nms5.json").read_text())["results"][key]
        op = res["best_f1_5s"]
        for c, a in res["per_class_ap_5s"].items():
            class_ap[c].append(a)
        fold_metrics[fd.name] = {**metrics(fd), "threshold": op["threshold"], f"{args.variant}:F1@5": op["f1"]}
        n_tp = n_fp = n_gt = 0
        for p in prep.patients:
            dets = [d for d in detections(prep, p, args.variant, args.alpha, args.phase_smooth, args.gate_floor)
                    if d[2] >= op["threshold"] and d[0] not in prep.bg]
            gts = [g for g in prep.gts[p] if g[0] not in prep.bg]
            labels = pd.read_csv(Path(args.frame_dir) / p / "Label" / f"{p}_all_labels.csv",
                                 usecols=["second", "frame_path"])
            # a few CSVs repeat a second: predictions follow the rows, times are the `second` values
            assert (labels["second"].to_numpy() == prep.preds[p]["second"]).all(), p
            ph_gt = prep.gt_phases[p]
            ph_hat = prep.preds[p]["probs_phases"].argmax(1)
            ph_true.append(ph_gt[: len(ph_hat)])
            ph_pred.append(ph_hat[: len(ph_gt)])
            for it in match_video(dets, gts, args.delta):
                n_tp += it["kind"] == "corretto"
                n_fp += it["kind"] != "corretto" and it["pred"] is not None
                n_gt += it["true"] is not None and it["kind"] != "duplicato"
                center = int(round(it["t_pred"] if it["t_pred"] is not None else it["t_gt"]))
                row = min(int(np.searchsorted(labels["second"].to_numpy(), center)), len(labels) - 1)
                it.update(patient=p, fold=fold, labels=labels, row=row)
                it["dir"] = out / "results" / folder_name(it, names) / f"{p}_t{center:05d}_fold{fold}"
                fmt = lambda v: "-" if v is None else f"{v:g}"  # noqa: E731
                it["info"] = (
                    f"video: {p} (fold {fold}, mai visto in training)\n"
                    f"tipo: {it['kind']}\n"
                    f"evento reale: {names[it['true']] if it['true'] is not None else NO_EVENT} @ {fmt(it['t_gt'])} s\n"
                    f"evento predetto: {names[it['pred']] if it['pred'] is not None else MISSED} @ {fmt(it['t_pred'])} s"
                    f"  score {fmt(None if it['score'] is None else round(it['score'], 3))}\n"
                    f"offset predetto-reale: "
                    f"{fmt(None if it['t_gt'] is None or it['t_pred'] is None else it['t_pred'] - it['t_gt'])} s\n"
                    f"fase reale @ {center}s: {ph_names[ph_gt[row]]}\n"
                    f"fase predetta @ {center}s: {ph_names[ph_hat[row]]}\n"
                    f"clip encoder: {meta_clip(prep.meta)}\n"
                )
                items.append(it)
        checks.append(dict(fold=fd.name, tp=n_tp, fp=n_fp, n_gt=n_gt,
                           ok=(n_tp, n_fp, n_gt) == (op["tp"], op["fp"], op["n_gt"]), report=op))
        print(f"{fd.name}: tp {n_tp}/{op['tp']} fp {n_fp}/{op['fp']} gt {n_gt}/{op['n_gt']} th {op['threshold']}",
              flush=True)
    assert all(c["ok"] for c in checks), "counts differ from eval_constraints: see summary.json"

    # ---- confusion matrices -------------------------------------------------------
    ev = [i for i, n in enumerate(names) if n.lower() != "idle"]
    rows = [names[i] for i in ev] + [NO_EVENT]
    cols = [names[i] for i in ev] + [MISSED]
    r_of = {c: k for k, c in enumerate(ev)}
    mat = np.zeros((len(rows), len(cols)), dtype=int)
    for it in items:
        r = len(ev) if it["kind"] in ("falso_allarme", "duplicato") else r_of[it["true"]]
        c = len(ev) if it["pred"] is None else r_of[it["pred"]]
        mat[r, c] += 1
    assert np.trace(mat[: len(ev), : len(ev)]) == sum(c["tp"] for c in checks)
    plot_matrix(mat, rows, cols, out / "confusion_matrix.png",
                f"Eventi, {args.variant} @ miglior F1, ±{args.delta:g} s, {len(fold_dirs)} fold "
                f"(riga {NO_EVENT} = falsi allarmi e duplicati)")
    ph_t, ph_p = np.concatenate(ph_true), np.concatenate(ph_pred)
    ph_mat = np.zeros((len(ph_names), len(ph_names)), dtype=int)
    np.add.at(ph_mat, (ph_t, ph_p), 1)
    plot_matrix(ph_mat, ph_names, ph_names, out / "confusion_matrix_phases.png",
                f"Fasi per secondo (argmax), {len(fold_dirs)} fold, accuratezza {np.trace(ph_mat) / ph_mat.sum():.3f}",
                annotate=False)

    # ---- frames and html ----------------------------------------------------------
    by_pair = defaultdict(list)
    for it in items:
        by_pair[it["dir"].parent.name].append(it)
    if not args.no_frames:
        clip = {fd.name.removeprefix("fold"): json.loads((fd / "val" / "meta.json").read_text()) for fd in fold_dirs}
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(lambda it: export_item(it, args.frame_dir, clip[it["fold"]]["num_frames"],
                                               clip[it["fold"]]["sampling_rate"], clip[it["fold"]]["seq_mode"]),
                        items))
        for pair, its in by_pair.items():
            write_pair_html(out / "results" / pair, its)
    kinds_of = {p: its[0]["kind"] for p, its in by_pair.items()}
    pairs = sorted(((p, len(v)) for p, v in by_pair.items()), key=lambda x: -x[1])
    write_html(out, pairs, kinds_of, len(items), out.name)

    # ---- summary ------------------------------------------------------------------
    keys = sorted(set.intersection(*(set(m) for m in fold_metrics.values())))
    summary = {
        "temporal_root": str(Path(args.temporal_root).resolve()),
        "decoder": key,
        "delta": args.delta,
        "folds": fold_metrics,
        "mean_std": {k: [float(np.mean([m[k] for m in fold_metrics.values()])),
                         float(np.std([m[k] for m in fold_metrics.values()]))] for k in keys},
        "kinds": dict(Counter(it["kind"] for it in items)),
        "pairs": dict(sorted(((p, len(v)) for p, v in by_pair.items()), key=lambda x: -x[1])),
        "per_class_ap5": {c: [float(np.mean(v)), float(np.std(v))] for c, v in class_ap.items()},
        "phase_accuracy_pooled": float(np.trace(ph_mat) / ph_mat.sum()),
        "checks": [{k: v for k, v in c.items() if k != "report"} for c in checks],
        "confusion": {"rows": rows, "cols": cols, "matrix": mat.tolist()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["kinds"]), f"{len(by_pair)} cartelle", flush=True)


def meta_clip(meta):
    return f"{meta['num_frames']} frame a 1 fps, rate {meta['sampling_rate']}, modo {meta['seq_mode']}"


if __name__ == "__main__":
    main()
