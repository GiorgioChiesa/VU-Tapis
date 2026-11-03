"""
Segment-level metrics for dense per-second labels (phases), as in MS-TCN / ASFormer.

* edit score (Lea et al., 2017): 100 * (1 - normalised Levenshtein distance) between
  the ordered segment label sequences, averaged over videos;
* segmental F1@k: a predicted segment is a true positive when its IoU with an unmatched
  ground-truth segment of the same class is >= k / 100; tp/fp/fn are summed over videos
  before computing F1 (MS-TCN `eval.py`).

Both reward the absence of over-segmentation, which frame accuracy does not see.
"""

import numpy as np

OVERLAPS = (10, 25, 50)


def segments(labels, ignore=()):
    """
    Run-length segments ``(label, start, end)`` with ``end`` exclusive.

    >>> segments([0, 0, 1, 1, 1, 0, 2])
    [(0, 0, 2), (1, 2, 5), (0, 5, 6), (2, 6, 7)]
    >>> segments([0, 0, 1, 1, 1, 0, 2], ignore=(0,))
    [(1, 2, 5), (2, 6, 7)]
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(labels)]
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends) if labels[s] not in ignore]


def levenshtein(a, b):
    """
    >>> levenshtein([1, 2, 3], [1, 3])
    1
    >>> levenshtein([], [4, 5])
    2
    """
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit_score(pred, gt, ignore=()):
    """
    >>> edit_score([0, 0, 1, 1, 2], [0, 1, 1, 1, 2])
    100.0
    >>> edit_score([0, 1, 0, 1, 2], [0, 1, 1, 1, 2])  # over-segmentation: 5 segments vs 3
    60.0
    """
    p = [s[0] for s in segments(pred, ignore)]
    g = [s[0] for s in segments(gt, ignore)]
    if not p and not g:
        return 100.0
    return float((1 - levenshtein(p, g) / max(len(p), len(g))) * 100)


def segment_counts(pred, gt, overlap, ignore=()):
    """
    ``(tp, fp, fn)`` at IoU threshold ``overlap`` (fraction).

    >>> segment_counts([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], 0.5)
    (2, 0, 0)
    >>> segment_counts([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], 0.9)
    (0, 2, 2)
    >>> segment_counts([1, 0, 1, 1], [1, 1, 1, 1], 0.5)  # split segment: one tp, one duplicate
    (1, 2, 0)
    """
    ps, gs = segments(pred, ignore), segments(gt, ignore)
    used = np.zeros(len(gs), dtype=bool)
    tp = fp = 0
    for c, s, e in ps:
        best, best_j = -1.0, -1
        for j, (gc, gs_, ge) in enumerate(gs):
            if gc != c:
                continue
            inter = max(0, min(e, ge) - max(s, gs_))
            iou = inter / (max(e, ge) - min(s, gs_))
            if iou > best:
                best, best_j = iou, j
        if best >= overlap and not used[best_j]:
            tp += 1
            used[best_j] = True
        else:
            fp += 1
    return tp, fp, int((~used).sum())


def changes(labels):
    """
    Seconds where the label changes.

    >>> changes([0, 0, 1, 1, 2]).tolist()
    [2, 4]
    """
    labels = np.asarray(labels)
    return np.flatnonzero(labels[1:] != labels[:-1]) + 1


def boundary_counts(pred, gt, tol):
    """
    ``(tp, fp, fn)``: changes of ``pred`` matched one-to-one to changes of ``gt`` within ``tol``
    seconds, closest pairs first. The class on either side of a change is not checked.

    >>> boundary_counts([0, 0, 1, 1, 1, 2], [0, 0, 0, 1, 1, 2], 1)
    (2, 0, 0)
    >>> boundary_counts([0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], 1)  # flicker: one tp, two fp
    (1, 2, 0)
    >>> boundary_counts([0, 0, 0, 0], [0, 0, 1, 1], 5)
    (0, 0, 1)
    """
    p, g = changes(pred), changes(gt)
    dist = np.abs(p[:, None] - g[None, :])
    pairs = sorted((int(dist[i, j]), i, j) for i, j in zip(*np.nonzero(dist <= tol)))
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_p), len(p) - len(used_p), len(g) - len(used_g)


def boundary_f1(preds, gts, tol):
    """
    Boundary precision / recall / F1 (fractions, not percent) at ``tol`` s, counts summed over videos.

    >>> boundary_f1({"v": [0, 1, 0, 1, 1, 1]}, {"v": [0, 0, 0, 1, 1, 1]}, 1)
    {'P': 0.3333333333333333, 'R': 1.0, 'F1': 0.5}
    """
    tp = fp = fn = 0
    for v in gts:
        a, b, c = boundary_counts(preds[v], gts[v], tol)
        tp, fp, fn = tp + a, fp + b, fn + c
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"P": prec, "R": rec, "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def segmental_metrics(preds, gts, overlaps=OVERLAPS, ignore=()):
    """
    Aggregate over videos: ``{"edit", "F1@10", "F1@25", "F1@50"}`` (percent).

    >>> m = segmental_metrics({"v": [0, 0, 1, 1]}, {"v": [0, 0, 1, 1]})
    >>> m["edit"], m["F1@50"]
    (100.0, 100.0)
    >>> round(segmental_metrics({"v": [0, 0, 0, 0]}, {"v": [0, 0, 1, 1]})["F1@25"], 2)  # P = 1, R = 1/2
    66.67
    """
    out = {"edit": float(np.mean([edit_score(preds[v], gts[v], ignore) for v in gts]))}
    for k in overlaps:
        tp = fp = fn = 0
        for v in gts:
            a, b, c = segment_counts(preds[v], gts[v], k / 100, ignore)
            tp, fp, fn = tp + a, fp + b, fn + c
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[f"F1@{k}"] = float(2 * prec * rec / (prec + rec) * 100) if prec + rec else 0.0
    return out
