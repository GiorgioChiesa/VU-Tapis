"""
Event spotting evaluation on untrimmed videos.

An event is a point in time ``(class, t)``. A prediction ``(class, t, score)`` is a
true positive if it can be matched one-to-one to a ground-truth event of the same
class with ``|t_pred - t_gt| <= delta`` (delta is a +/- tolerance in seconds).
Matching is greedy by descending score, inside each video. AP is computed per
class by pooling the detections of all videos (all-point interpolated precision,
as in PASCAL VOC / SoccerNet).

Everything here is pure numpy so it can be unit-tested without a model.
"""

from collections import defaultdict

import numpy as np

TIGHT_DELTAS = (1, 2, 3, 4, 5)
LOOSE_DELTAS = (5, 10, 15, 20, 25, 30)


def peak_detection(scores, nms_window, threshold=0.0):
    """
    Greedy 1-D non-maximum suppression on a per-second score curve.

    Args:
        scores (np.ndarray): shape ``[T]``, score of one class for each second.
        nms_window (float): half window in seconds; after picking a peak at ``t``
            every second in ``[t - nms_window, t + nms_window]`` is suppressed.
        threshold (float): peaks with score below this value are discarded.

    Returns:
        list of ``(t, score)`` sorted by time.

    >>> peak_detection(np.array([0.1, 0.9, 0.8, 0.1, 0.0, 0.7]), nms_window=1, threshold=0.5)
    [(1, 0.9), (5, 0.7)]
    """
    scores = np.asarray(scores, dtype=np.float64).copy()
    peaks = []
    if scores.size == 0:
        return peaks
    window = int(np.floor(nms_window))
    while True:
        t = int(np.argmax(scores))
        s = scores[t]
        if not np.isfinite(s) or s < threshold or s <= 0:
            break
        peaks.append((t, float(s)))
        scores[max(0, t - window) : t + window + 1] = -np.inf
    return sorted(peaks)


def decode_events(probs, nms_window=5, threshold=0.0, background_classes=(0,), mode="nms"):
    """
    Turn per-second class probabilities into point detections.

    Args:
        probs (np.ndarray): ``[T, C]`` probabilities for one video.
        nms_window (float): half window of the NMS, in seconds (mode ``nms``).
        threshold (float): minimum score of a detection.
        background_classes (tuple): classes that never produce detections (Idle).
        mode (str): ``nms`` runs peak detection independently for every class.
            ``argmax`` emits one detection for every second whose argmax is a
            foreground class (no suppression): it reproduces a naive frame-wise
            classifier and shows how many duplicates / false alarms it produces.

    Returns:
        list of ``(class, t, score)``.
    """
    probs = np.asarray(probs)
    detections = []
    if mode == "argmax":
        top = probs.argmax(axis=1)
        for t, c in enumerate(top):
            if c in background_classes or probs[t, c] < threshold:
                continue
            detections.append((int(c), t, float(probs[t, c])))
        return detections
    if mode != "nms":
        raise ValueError(f"Unknown decoding mode {mode}")
    for c in range(probs.shape[1]):
        if c in background_classes:
            continue
        for t, s in peak_detection(probs[:, c], nms_window, threshold):
            detections.append((c, t, s))
    return detections


def match_events(pred_times, pred_scores, gt_times, delta):
    """
    Greedy one-to-one matching of predictions to ground truth (one class, one video).

    Predictions are visited by descending score; each takes the closest still
    unmatched ground-truth event within ``delta``.

    Returns:
        tp (np.ndarray[bool]): for each prediction (in input order) whether it is a TP.
        offsets (np.ndarray): ``t_pred - t_gt`` for every TP (in input order of TPs).
        matched_gt (np.ndarray[bool]): for each ground-truth event whether it was found.

    >>> tp, off, m = match_events([10, 11, 30], [0.9, 0.8, 0.5], [10, 50], delta=2)
    >>> tp.tolist(), off.tolist(), m.tolist()
    ([True, False, False], [0.0], [True, False])
    """
    pred_times = np.asarray(pred_times, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_times = np.asarray(gt_times, dtype=np.float64)
    tp = np.zeros(len(pred_times), dtype=bool)
    offsets = np.full(len(pred_times), np.nan)
    matched_gt = np.zeros(len(gt_times), dtype=bool)
    for i in np.argsort(-pred_scores, kind="stable"):
        if len(gt_times) == 0:
            break
        dist = np.abs(gt_times - pred_times[i])
        dist[matched_gt] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= delta:
            tp[i] = True
            matched_gt[j] = True
            offsets[i] = pred_times[i] - gt_times[j]
    return tp, offsets[tp], matched_gt


def average_precision(scores, tp, n_gt):
    """
    All-point interpolated AP from pooled detections of one class.

    >>> average_precision([0.9, 0.8, 0.7], [True, False, True], n_gt=2)
    0.8333333333333333
    """
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    tp = np.asarray(tp, dtype=np.float64)[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _group(events, with_score):
    """``{video: [(c, t[, s])]}`` -> ``{(video, c): (times, scores)}``."""
    out = defaultdict(lambda: ([], []))
    for video, evs in events.items():
        for e in evs:
            times, scores = out[(video, int(e[0]))]
            times.append(float(e[1]))
            scores.append(float(e[2]) if with_score else 1.0)
    return out


def spotting_map(preds, gts, deltas, num_classes, ignore_classes=(0,)):
    """
    Spotting mAP for several tolerances.

    Args:
        preds (dict): ``{video: [(class, t, score), ...]}``.
        gts (dict): ``{video: [(class, t), ...]}``. Videos missing from ``preds``
            count as having no detections.
        deltas (iterable): tolerances (+/- seconds).
        num_classes (int): number of classes, including the ignored ones.
        ignore_classes (tuple): classes left out of the mean (Idle).

    Returns:
        dict with ``ap[delta][class]`` (nan for classes without ground truth),
        ``map[delta]`` (mean over classes with ground truth) and ``n_gt[class]``.
    """
    pred_g = _group(preds, with_score=True)
    gt_g = _group(gts, with_score=False)
    videos = set(gts) | set(preds)
    classes = [c for c in range(num_classes) if c not in ignore_classes]
    n_gt = {c: sum(len(gt_g[(v, c)][0]) for v in videos if (v, c) in gt_g) for c in classes}
    result = {"ap": {}, "map": {}, "n_gt": n_gt}
    for delta in deltas:
        ap = {}
        for c in classes:
            all_scores, all_tp = [], []
            for v in videos:
                p_t, p_s = pred_g[(v, c)] if (v, c) in pred_g else ([], [])
                g_t = gt_g[(v, c)][0] if (v, c) in gt_g else []
                tp, _, _ = match_events(p_t, p_s, g_t, delta)
                all_scores.extend(p_s)
                all_tp.extend(tp.tolist())
            ap[c] = average_precision(all_scores, all_tp, n_gt[c])
        valid = [a for c, a in ap.items() if n_gt[c] > 0]
        result["ap"][delta] = ap
        result["map"][delta] = float(np.mean(valid)) if valid else float("nan")
    return result


def average_map(map_by_delta, deltas):
    """Mean of ``map[delta]`` over ``deltas`` (SoccerNet-style average-mAP)."""
    return float(np.mean([map_by_delta[d] for d in deltas]))


def operating_point(preds, gts, delta, durations, threshold=0.0, ignore_classes=(0,)):
    """
    Event-level metrics at a fixed score threshold.

    Args:
        preds, gts: as in :func:`spotting_map`.
        delta (float): tolerance in seconds.
        durations (dict): ``{video: seconds}``, used for false alarms per hour.
        threshold (float): detections below this score are dropped.

    Returns:
        dict with precision, recall, f1, false alarms per hour, duplicates per
        ground-truth event (unmatched detections of the right class within delta
        of an already-found event) and timing errors of the matched detections.
    """
    pred_g = _group(
        {v: [e for e in evs if e[2] >= threshold and e[0] not in ignore_classes] for v, evs in preds.items()},
        with_score=True,
    )
    gt_g = _group({v: [e for e in evs if e[0] not in ignore_classes] for v, evs in gts.items()}, with_score=False)
    keys = set(pred_g) | set(gt_g)
    n_tp = n_fp = n_gt = n_dup = 0
    offsets = []
    for key in keys:
        p_t, p_s = pred_g[key] if key in pred_g else ([], [])
        g_t = gt_g[key][0] if key in gt_g else []
        tp, off, matched = match_events(p_t, p_s, g_t, delta)
        n_tp += int(tp.sum())
        n_fp += int((~tp).sum())
        n_gt += len(g_t)
        offsets.extend(off.tolist())
        if len(g_t) and (~tp).any():
            g = np.asarray(g_t)[matched]
            for t in np.asarray(p_t)[~tp]:
                if g.size and np.min(np.abs(g - t)) <= delta:
                    n_dup += 1
    hours = sum(durations.values()) / 3600.0
    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    offsets = np.asarray(offsets)
    return {
        "delta": delta,
        "threshold": threshold,
        "n_gt": n_gt,
        "n_pred": n_tp + n_fp,
        "tp": n_tp,
        "fp": n_fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_hour": n_fp / hours if hours > 0 else float("nan"),
        "duplicates_per_gt": n_dup / max(n_gt, 1),
        "mean_abs_error_s": float(np.mean(np.abs(offsets))) if offsets.size else float("nan"),
        "median_signed_error_s": float(np.median(offsets)) if offsets.size else float("nan"),
    }


def best_f1_threshold(preds, gts, delta, durations, thresholds, ignore_classes=(0,)):
    """Sweep a global threshold and return the operating point with the highest F1."""
    best = None
    for th in thresholds:
        op = operating_point(preds, gts, delta, durations, th, ignore_classes)
        if best is None or op["f1"] > best["f1"]:
            best = op
    return best
