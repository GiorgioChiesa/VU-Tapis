#!/usr/bin/env python3
"""
Experiment 7: per-video phase segmentation metrics and the shared phase-posterior format.

Posterior format (read by Esp6 `tools/phase_decode.py`): ``<run_dir>/phase_posteriors/<video>.npz`` with
``probs`` float32 [T, C] (softmax, 1 fps), ``gt`` int [T] (local phase ids), ``fold`` int; plus
``<run_dir>/phase_posteriors/classes.json`` (list of the C phase names, index = local id).

Per video (labels = argmax of the posteriors):
  acc, macro_F1 and jaccard (mean over the classes present in the ground truth or the prediction of that
  video), edit, seg F1@10/25/50 (segment_eval counts of that video), boundary F1 +/- 5 s, seg_ratio
  (predicted / true segments). A run is summarised by the mean over its videos; the pooled quantities
  (``pooled_*``: segment counts summed over videos as in MS-TCN, changes ratio, per-phase frame F1 over
  all frames of the run) are kept for comparison with older notes.

>>> m = video_metrics(np.array([0, 0, 1, 1, 1, 2]), np.array([0, 0, 1, 1, 2, 2]))
>>> round(m["acc"], 3), round(m["macro_F1"], 3), m["edit"], m["F1@50"], m["boundary_F1@5s"], m["seg_ratio"]
(0.833, 0.822, 100.0, 100.0, 1.0, 1.0)
>>> video_metrics(np.array([0, 1, 0, 1]), np.array([0, 0, 0, 0]))["seg_ratio"]
4.0
"""

import json
from pathlib import Path

import numpy as np

from tapis.evaluate import segment_eval

OVERLAPS = (10, 25, 50)
BOUNDARY_TOL = 5
PER_VIDEO = ["acc", "macro_F1", "jaccard", "edit", "F1@10", "F1@25", "F1@50", "boundary_F1@5s", "seg_ratio"]


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def video_metrics(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    classes = np.union1d(np.unique(gt), np.unique(pred))
    f1s, ious = [], []
    for c in classes:
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        f1s.append(f1(tp, fp, fn))
        ious.append(tp / (tp + fp + fn))
    m = {"acc": float((pred == gt).mean()), "macro_F1": float(np.mean(f1s)), "jaccard": float(np.mean(ious)),
         "edit": segment_eval.edit_score(pred, gt)}
    for k in OVERLAPS:
        m[f"F1@{k}"] = 100 * f1(*segment_eval.segment_counts(pred, gt, k / 100))
    m["boundary_F1@5s"] = f1(*segment_eval.boundary_counts(pred, gt, BOUNDARY_TOL))
    m["seg_ratio"] = len(segment_eval.segments(pred)) / len(segment_eval.segments(gt))
    return m


def run_metrics(labels, gts, n_classes):
    """
    ``labels``/``gts``: {video: int array}. Mean over videos of `video_metrics`, plus pooled metrics.

    >>> r = run_metrics({"a": np.array([0, 0, 1]), "b": np.array([1, 1, 1])}, {"a": np.array([0, 0, 1]), "b": np.array([1, 1, 0])}, 2)
    >>> round(r["acc"], 3), r["n_videos"], r["pooled_changes_ratio"]
    (0.833, 2, 0.5)
    """
    per = {v: video_metrics(labels[v], gts[v]) for v in gts}
    out = {k: float(np.mean([m[k] for m in per.values()])) for k in PER_VIDEO}
    out["n_videos"] = len(per)
    pooled = segment_eval.segmental_metrics(labels, gts)
    out.update({f"pooled_{k}": v for k, v in pooled.items()})
    n_pred = sum(len(segment_eval.changes(labels[v])) for v in gts)
    n_true = sum(len(segment_eval.changes(gts[v])) for v in gts)
    out["pooled_changes_ratio"] = n_pred / max(1, n_true)
    p, g = np.concatenate([labels[v] for v in gts]), np.concatenate([gts[v] for v in gts])
    per_class = []
    for c in range(n_classes):
        tp, fp, fn = int(((p == c) & (g == c)).sum()), int(((p == c) & (g != c)).sum()), int(((p != c) & (g == c)).sum())
        per_class.append(f1(tp, fp, fn) if tp + fp + fn else float("nan"))
    out["per_phase_F1"] = per_class
    out["pooled_macro_F1"] = float(np.nanmean(per_class))
    out["pooled_acc"] = float((p == g).mean())
    return out


def write_posteriors(run_dir, probs, gts, fold, classes, sub="phase_posteriors"):
    """``probs``/``gts``: {video: array}. Writes the shared posterior format under ``run_dir/sub``."""
    d = Path(run_dir) / sub
    d.mkdir(parents=True, exist_ok=True)
    for v, pr in probs.items():
        np.savez_compressed(d / f"{v}.npz", probs=pr.astype(np.float32), gt=np.asarray(gts[v]).astype(np.int64),
                            fold=np.int64(fold))
    (d / "classes.json").write_text(json.dumps(list(classes)))
    return d


def read_posteriors(d):
    """``(probs, gts, fold)`` dicts of a posterior directory."""
    probs, gts, folds = {}, {}, {}
    for z in sorted(Path(d).glob("*.npz")):
        f = np.load(z)
        probs[z.stem], gts[z.stem], folds[z.stem] = f["probs"], f["gt"], int(f["fold"])
    return probs, gts, folds


def posterior_metrics(d, n_classes=None):
    probs, gts, _ = read_posteriors(d)
    n_classes = n_classes or next(iter(probs.values())).shape[1]
    return run_metrics({v: p.argmax(1) for v, p in probs.items()}, gts, n_classes)
