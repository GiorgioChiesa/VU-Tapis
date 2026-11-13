#!/usr/bin/env python3
"""
Decoding of per-second surgical phase posteriors into flicker-free phase segments (Esp6).

Input is the shared posterior format: ``<run>/phase_posteriors/<video>.npz`` with ``probs``
float32 [T, C] (softmax, 1 fps), ``gt`` int [T] (local ids, -1 = unlabelled), ``fold`` int, plus
``<run>/phase_posteriors/classes.json`` (local id -> name). ``tools/export_phase_posteriors.py``
writes it for train_hybrid.py runs.

Decoders (``DECODERS``; every one is ``f(probs, priors, **params) -> labels [T]``):
  * ``argmax``      P0, raw argmax;
  * ``moving_avg``  P0, argmax of a ``window`` s moving average of the posteriors;
  * ``hysteresis``  P1, switch only when the new argmax beats the current class by ``margin``, then
                    merge segments shorter than a per-phase minimum duration
                    ``d_c = factor * percentile_q(train durations of phase c)`` (or one global ``min_dur``);
  * ``hsmm``        P2, explicit-duration hidden semi-Markov Viterbi on ``bin`` s blocks, log-normal
                    duration priors per phase, transitions ``learned`` (smoothed counts, every
                    transition allowed), ``order`` (forward along the modal order only) or ``free``
                    (uniform, durations only); boundaries then refined at 1 s;
  * ``pelt``        P3, PELT change points on the posteriors (L2 cost, ``penalty``, ``min_size``) and
                    the argmax of the mean posterior in each segment.

Priors (``fit_priors``) are estimated from ground-truth label sequences of training videos only.

CLI (leave-fold-out priors: each video is decoded with priors from the videos of the other folds
of the same directory):
    python tools/phase_decode.py --post-dir <run>/phase_posteriors --decoder hsmm \
        --param mode=learned --out <dir>        # -> <dir>/labels/<video>.npy, <dir>/metrics.json
    Per-phase values are comma separated (``--decoder hysteresis --param margin=0.1,0.3,... min_dur=20,60,...``).
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

EPS = 1e-8


# ---- I/O ---------------------------------------------------------------------------
def load_posteriors(post_dir):
    """``({video: {"probs", "gt", "fold"}}, class names)`` of a ``phase_posteriors`` directory."""
    post_dir = Path(post_dir)
    names = json.loads((post_dir / "classes.json").read_text())
    names = [names[str(i)] for i in range(len(names))] if isinstance(names, dict) else list(names)
    videos = {}
    for f in sorted(post_dir.glob("*.npz")):
        z = np.load(f)
        videos[f.stem] = {"probs": z["probs"].astype(np.float32), "gt": z["gt"].astype(int), "fold": int(z["fold"])}
    return videos, names


# ---- segments and priors -----------------------------------------------------------
def runs(labels):
    """
    Run-length segments ``(label, start, end)``, ``end`` exclusive.

    >>> runs([0, 0, 1, 1, 1, 0])
    [(0, 0, 2), (1, 2, 5), (0, 5, 6)]
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(labels)]
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def modal_order(sequences, n_classes):
    """
    Phases sorted by their median relative start position over the videos where they occur, ties
    broken by the mean position (absent phases last). Orsi: Right and Left lateral dissection both
    have median 0.5; Left comes first in 30/50 videos and has the lower mean.

    >>> modal_order([[0, 2, 1], [0, 2, 1, 2]], 3)
    [0, 2, 1]
    >>> modal_order([[0, 2, 1, 3, 4], [0, 1, 2, 3, 4], [0, 2, 1, 3, 4]], 5)
    [0, 2, 1, 3, 4]
    """
    pos = {c: [] for c in range(n_classes)}
    for seq in sequences:
        n = len(seq)
        seen = set()
        for i, c in enumerate(seq):
            if c not in seen:
                pos[c].append(i / max(1, n - 1))
                seen.add(c)
    return sorted(range(n_classes), key=lambda c: (np.median(pos[c]), np.mean(pos[c]), c) if pos[c] else (2.0, 2.0, c))


@dataclass
class Priors:
    n_classes: int
    durations: list  # per class: array of segment durations (s)
    trans: np.ndarray  # [C, C] transition counts between consecutive segments
    start: np.ndarray  # [C] counts of the first segment
    order: list  # modal order of the phases
    sequences: list = field(default_factory=list)


def fit_priors(gts, n_classes):
    """
    Duration, transition, start and order statistics of ground-truth label sequences (training videos).
    Unlabelled seconds (-1) are dropped.

    >>> p = fit_priors([np.array([0, 0, 1, 1, 1, 2]), np.array([0, 1, 1, 2, 2])], 3)
    >>> [d.tolist() for d in p.durations], p.trans[1, 2], p.start.tolist(), p.order
    ([[2, 1], [3, 2], [1, 2]], 2.0, [2.0, 0.0, 0.0], [0, 1, 2])
    """
    durations = [[] for _ in range(n_classes)]
    trans = np.zeros((n_classes, n_classes))
    start = np.zeros(n_classes)
    seqs = []
    for g in gts:
        g = np.asarray(g)
        g = g[g >= 0]
        seg = runs(g)
        if not seg:
            continue
        seqs.append([c for c, _, _ in seg])
        start[seg[0][0]] += 1
        for c, s, e in seg:
            durations[c].append(e - s)
        for (a, _, _), (b, _, _) in zip(seg[:-1], seg[1:]):
            trans[a, b] += 1
    return Priors(n_classes, [np.asarray(d, dtype=int) for d in durations], trans, start,
                  modal_order(seqs, n_classes), seqs)


def min_durations(priors, q=5.0, factor=0.5, floor=1.0):
    """
    Per-phase minimum duration ``factor * percentile_q(train durations)`` (s; ``factor`` scalar or per
    phase); phases never seen in training get the median percentile over the other phases.

    >>> p = fit_priors([np.array([0] * 10 + [1] * 40)], 3)
    >>> min_durations(p, q=50, factor=0.5).tolist(), min_durations(p, q=50, factor=[1, 0.5, 2]).tolist()
    ([5.0, 20.0, 12.5], [10.0, 20.0, 50.0])
    """
    d = np.array([np.percentile(x, q) if len(x) else np.nan for x in priors.durations])
    d[np.isnan(d)] = np.nanmedian(d) if np.isfinite(d).any() else floor
    d = d * np.broadcast_to(np.asarray(factor, dtype=float), d.shape)  # factor: scalar or per phase
    return np.maximum(d, floor)


# ---- P0 ----------------------------------------------------------------------------
def decode_argmax(probs, priors=None):
    return np.asarray(probs).argmax(1)


def decode_moving_avg(probs, priors=None, window=15):
    """
    >>> decode_moving_avg(np.eye(2)[[0, 0, 1, 0, 0, 1, 1, 1]], window=3).tolist()
    [0, 0, 0, 0, 0, 1, 1, 1]
    """
    return uniform_filter1d(np.asarray(probs, dtype=np.float64), int(window), axis=0, mode="nearest").argmax(1)


# ---- P1 ----------------------------------------------------------------------------
def hysteresis(probs, margin):
    """
    Keep the current class until another class beats it by ``margin``: a scalar, or one value per
    class (the margin needed to switch INTO that class).

    >>> p = np.array([[.9, .1], [.45, .55], [.3, .7], [.6, .4]])
    >>> hysteresis(p, 0.2).tolist(), hysteresis(p, 0.0).tolist(), hysteresis(p, [0.0, 0.5]).tolist()
    ([0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0])
    """
    probs = np.asarray(probs)
    am = probs.argmax(1)
    mg = np.broadcast_to(np.asarray(margin, dtype=float), (probs.shape[1],)).tolist()
    pl, am_l = probs.tolist(), am.tolist()  # plain lists: the loop is sequential
    out = [0] * len(pl)
    c = am_l[0]
    for t, (k, row) in enumerate(zip(am_l, pl)):
        if k != c and row[k] >= row[c] + mg[k]:
            c = k
        out[t] = c
    return np.asarray(out, dtype=int)


def merge_short(labels, probs, min_dur):
    """
    Repeatedly absorb the segment that is shortest relative to its class minimum duration
    (``min_dur`` per class, or a scalar) into the neighbour whose class has the higher mean
    posterior over it, until every segment is long enough (or one segment is left).

    >>> p = np.eye(3)[[0, 0, 0, 0, 2, 1, 1, 1, 1]] * .8 + .1
    >>> merge_short(np.array([0, 0, 0, 0, 2, 1, 1, 1, 1]), p, 2).tolist()
    [0, 0, 0, 0, 0, 1, 1, 1, 1]
    """
    labels = np.asarray(labels).copy()
    probs = np.asarray(probs)
    md = np.broadcast_to(np.asarray(min_dur, dtype=float), (probs.shape[1],))
    cs = np.vstack([np.zeros((1, probs.shape[1])), np.cumsum(probs, 0)])
    seg = [list(s) for s in runs(labels)]
    while len(seg) > 1:
        ratio = [(e - s) / md[c] for c, s, e in seg]
        i = int(np.argmin(ratio))
        if ratio[i] >= 1:
            break
        c, s, e = seg[i]
        cand = [j for j in (i - 1, i + 1) if 0 <= j < len(seg)]
        mean = lambda k: (cs[e, k] - cs[s, k]) / (e - s)
        j = max(cand, key=lambda j: mean(seg[j][0]))
        k = seg[j][0]
        seg[i][0] = k
        # merge runs of equal class around i
        lo, hi = i, i
        while lo > 0 and seg[lo - 1][0] == k:
            lo -= 1
        while hi < len(seg) - 1 and seg[hi + 1][0] == k:
            hi += 1
        seg[lo:hi + 1] = [[k, seg[lo][1], seg[hi][2]]]
    out = np.empty(len(labels), dtype=int)
    for c, s, e in seg:
        out[s:e] = c
    return out


def decode_hysteresis(probs, priors=None, margin=0.1, q=5.0, factor=0.5, min_dur=None):
    """
    P1: hysteresis (``margin``: scalar or per phase), then merge of short segments with a per-phase
    minimum duration: ``factor * percentile_q`` of the training durations (from ``priors``), or
    ``min_dur`` (scalar or per phase, s) when given.
    """
    lab = hysteresis(probs, margin)
    md = min_dur if min_dur is not None else min_durations(priors, q, factor)
    return merge_short(lab, probs, md)


# ---- P2: hidden semi-Markov Viterbi -------------------------------------------------
def lognormal_fit(d, sigma_floor=0.3):
    """``(mu, sigma)`` of log durations; None when there is no data."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return None
    lg = np.log(np.maximum(d, 1.0))
    sd = lg.std(ddof=1) if len(lg) > 1 else 1.0
    return float(lg.mean()), float(max(sd, sigma_floor))


def transition_logprob(priors, mode="learned", alpha=0.5):
    """
    Log transition matrix (diagonal -inf: segments) and log start probabilities.

    ``learned``: counts + ``alpha`` on every off-diagonal cell; ``order``: same counts restricted to
    forward moves along the modal order; ``free``: uniform.

    >>> p = fit_priors([np.array([0, 1, 2]), np.array([0, 2, 1])], 3)
    >>> A, S = transition_logprob(p, "order", alpha=0.5)
    >>> np.isfinite(A).astype(int).tolist(), round(float(np.exp(A[0, 1])), 3)
    ([[0, 1, 1], [0, 0, 1], [0, 0, 0]], 0.5)
    """
    C = priors.n_classes
    off = ~np.eye(C, dtype=bool)
    if mode == "free":
        A = np.where(off, -math.log(max(1, C - 1)), -np.inf)
        return A, np.full(C, -math.log(C))
    allowed = off.copy()
    if mode == "order":
        rank = np.empty(C, dtype=int)
        rank[priors.order] = np.arange(C)
        allowed = rank[None, :] > rank[:, None]
    elif mode != "learned":
        raise ValueError(mode)
    cnt = np.where(allowed, priors.trans + alpha, 0.0)
    rows = cnt.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(allowed & (rows > 0), np.log(cnt / np.where(rows > 0, rows, 1)), -np.inf)
    S = np.log((priors.start + alpha) / (priors.start + alpha).sum())
    return A, S


def refine_boundaries(labels, logp, radius):
    """
    Move every change of ``labels`` within +-``radius`` s to the second that maximises the
    log-likelihood of the two adjacent classes (segments are kept non-empty).

    >>> lp = np.log(np.eye(2)[[0, 0, 0, 1, 1, 1]] * .8 + .1)
    >>> refine_boundaries(np.array([0, 0, 0, 0, 0, 1]), lp, 3).tolist()
    [0, 0, 0, 1, 1, 1]
    """
    labels = np.asarray(labels).copy()
    seg = runs(labels)
    if len(seg) < 2:
        return labels
    bounds = [s for _, s, _ in seg[1:]]
    for i, b in enumerate(bounds):
        a, c = seg[i][0], seg[i + 1][0]
        lo = max(seg[i][1] + 1, b - radius)
        hi = min(seg[i + 1][2] - 1, b + radius)
        if hi <= lo:
            continue
        # score(s) = sum_{lo<=t<s} logp[t,a] + sum_{s<=t<hi+1} logp[t,c]
        win = logp[lo:hi + 1]
        ca = np.r_[0, np.cumsum(win[:, a])]
        cc = np.r_[0, np.cumsum(win[:, c])]
        score = ca + (cc[-1] - cc)
        s = lo + int(np.argmax(score[: hi - lo + 1]))
        labels[lo:hi + 1] = np.where(np.arange(lo, hi + 1) < s, a, c)
        seg[i] = (a, seg[i][1], s)
        seg[i + 1] = (c, s, seg[i + 1][2])
    return labels


def decode_hsmm(probs, priors, mode="learned", bin=5, temp=1.0, dur_weight=1.0, trans_weight=1.0, alpha=0.5,
                max_factor=2.0, refine=True):
    """
    P2: Viterbi over segmentations of ``bin`` s blocks. Score of a segment of class k covering blocks
    [s, t): sum of per-second ``log p / temp`` + ``dur_weight * log LogNormal_k(duration)`` +
    ``trans_weight * log A[prev, k]`` (or log start). Maximum duration per class:
    ``max_factor`` x the longest training segment.

    >>> p = fit_priors([np.array([0] * 30 + [1] * 30)] * 3, 2)
    >>> pr = np.eye(2)[[0] * 30 + [1] * 30] * .8 + .1
    >>> pr[10:12] = [.2, .8]                                       # a 2 s flicker
    >>> lab = decode_hsmm(pr, p, bin=5); (lab[:30] == 0).all() and (lab[30:] == 1).all()
    True
    """
    probs = np.asarray(probs, dtype=np.float64)
    T, C = probs.shape
    logp = np.log(probs + EPS) / temp
    nb = int(math.ceil(T / bin))
    pad = nb * bin - T
    lb = np.vstack([logp, np.zeros((pad, C))]).reshape(nb, bin, C).sum(1)
    cum = np.vstack([np.zeros((1, C)), np.cumsum(lb, 0)])
    A, S = transition_logprob(priors, mode, alpha)
    A, S = trans_weight * A, trans_weight * S

    # duration log-prior per class and length in blocks
    longest = max([int(d.max()) for d in priors.durations if len(d)] + [bin])
    dmax_c = np.array([max(1, int(math.ceil(max_factor * (d.max() if len(d) else longest) / bin))) for d in priors.durations])
    D = int(min(nb, dmax_c.max()))
    dsec = np.arange(1, D + 1) * bin
    logD = np.full((D, C), -np.inf)
    for k in range(C):
        fit = lognormal_fit(priors.durations[k])
        mu, sg = fit if fit else (math.log(longest / 2), 1.5)
        x = np.log(dsec)
        ld = -0.5 * ((x - mu) / sg) ** 2 - math.log(sg) - x  # log-normal density (constants dropped)
        ld[np.arange(1, D + 1) > dmax_c[k]] = -np.inf
        logD[:, k] = dur_weight * ld

    V = np.full((nb + 1, C), -np.inf)
    back_d = np.zeros((nb + 1, C), dtype=np.int32)
    back_j = np.full((nb + 1, C), -1, dtype=np.int32)
    U = np.full((nb + 1, C), -np.inf)  # best score entering class k at block t
    Uj = np.full((nb + 1, C), -1, dtype=np.int32)
    U[0] = S
    for t in range(1, nb + 1):
        dd = np.arange(1, min(t, D) + 1)
        s = t - dd
        cand = U[s] + logD[: len(dd)] + (cum[t] - cum[s])  # [len(dd), C]
        best = np.argmax(cand, 0)
        V[t] = cand[best, np.arange(C)]
        back_d[t] = dd[best]
        back_j[t] = Uj[s[best], np.arange(C)]
        if t < nb:
            M = V[t][:, None] + A  # from j (rows) to k (cols)
            Uj[t] = np.argmax(M, 0)
            U[t] = M[Uj[t], np.arange(C)]
    # backtrack
    lab_b = np.empty(nb, dtype=int)
    t, k = nb, int(np.argmax(V[nb]))
    while t > 0:
        d = int(back_d[t, k])
        j = int(back_j[t, k])
        lab_b[t - d:t] = k
        t, k = t - d, j
    lab = np.repeat(lab_b, bin)[:T]
    return refine_boundaries(lab, np.log(probs + EPS), bin) if refine and bin > 1 else lab


# ---- P3: change points ------------------------------------------------------------
def pelt(x, penalty, min_size=1):
    """
    Optimal change points of ``x`` [T, D] under an L2 (piecewise-constant mean) cost plus
    ``penalty`` per segment (Killick et al., 2012). Returns the segment ends (last = T).

    >>> x = np.r_[np.zeros(20), np.ones(20) * 3][:, None]
    >>> pelt(x, 1.0, 2)
    [20, 40]
    """
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    c1 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    c2 = np.r_[0.0, np.cumsum((x ** 2).sum(1))]

    def cost(s, t):  # s array, t scalar
        n = t - s
        return c2[t] - c2[s] - ((c1[t] - c1[s]) ** 2).sum(1) / n

    F = np.full(T + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(T + 1, dtype=int)
    R = np.array([0])
    for t in range(1, T + 1):
        ok = R[t - R >= min_size]
        if len(ok):
            c = F[ok] + cost(ok, t)
            val = c + penalty
            i = int(np.argmin(val))
            F[t], last[t] = val[i], ok[i]
            # prune (K = 0; the L2 cost is sub-additive) candidates that can never be optimal again
            R = np.setdiff1d(R, ok[c > F[t]], assume_unique=True)
        if np.isfinite(F[t]):
            R = np.r_[R, t]
    ends, t = [], T
    while t > 0:
        ends.append(t)
        t = last[t]
    return ends[::-1]


def decode_pelt(probs, priors=None, penalty=10.0, min_size=10):
    """P3: PELT on the posteriors, then the argmax of the mean posterior of every segment."""
    probs = np.asarray(probs, dtype=np.float64)
    ends = pelt(probs, penalty, max(1, min(int(min_size), len(probs))))
    out = np.empty(len(probs), dtype=int)
    s = 0
    for e in ends:
        out[s:e] = probs[s:e].mean(0).argmax()
        s = e
    return out


DECODERS = {
    "argmax": decode_argmax,
    "moving_avg": decode_moving_avg,
    "hysteresis": decode_hysteresis,
    "hsmm": decode_hsmm,
    "pelt": decode_pelt,
}


def decode(name, probs, priors=None, **params):
    return DECODERS[name](probs, priors, **params)


# ---- metrics -----------------------------------------------------------------------
def _seg_counts(pred, gt, overlap, ignore=()):
    ps = [s for s in runs(pred) if s[0] not in ignore]
    gs = [s for s in runs(gt) if s[0] not in ignore]
    used = np.zeros(len(gs), dtype=bool)
    tp = fp = 0
    per = {}
    for c, s, e in ps:
        best, bj = -1.0, -1
        for j, (gc, g0, g1) in enumerate(gs):
            if gc != c:
                continue
            iou = max(0, min(e, g1) - max(s, g0)) / (max(e, g1) - min(s, g0))
            if iou > best:
                best, bj = iou, j
        hit = best >= overlap and not used[bj]
        if hit:
            used[bj] = True
        tp, fp = tp + hit, fp + (not hit)
        r = per.setdefault(c, [0, 0, 0])
        r[0 if hit else 1] += 1
    for j, (gc, _, _) in enumerate(gs):
        if not used[j]:
            per.setdefault(gc, [0, 0, 0])[2] += 1
    return tp, fp, int((~used).sum()), per


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def _edit(pred, gt, ignore=()):
    p = [c for c, _, _ in runs(pred) if c not in ignore]
    g = [c for c, _, _ in runs(gt) if c not in ignore]
    if not p and not g:
        return 100.0
    prev = list(range(len(g) + 1))
    for i, x in enumerate(p, 1):
        cur = [i]
        for j, y in enumerate(g, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return 100.0 * (1 - prev[-1] / max(len(p), len(g)))


def _boundary(pred, gt, tol):
    p = np.flatnonzero(np.diff(pred)) + 1
    g = np.flatnonzero(np.diff(gt)) + 1
    dist = np.abs(p[:, None] - g[None, :])
    pairs = sorted((int(dist[i, j]), i, j) for i, j in zip(*np.nonzero(dist <= tol)))
    up, ug = set(), set()
    for _, i, j in pairs:
        if i not in up and j not in ug:
            up.add(i)
            ug.add(j)
    return len(up), len(p) - len(up), len(g) - len(ug)


def video_metrics(pred, gt, n_classes, ignore_idle=False, overlaps=(10, 25, 50), tol=5):
    """
    Per-video phase metrics. Frame metrics (accuracy, macro-F1, Jaccard over the classes present in
    the ground truth or the prediction) and segment metrics (F1@k %, edit %, boundary F1 at +-``tol`` s,
    predicted / true segments). ``ignore_idle``: class 0 seconds dropped from frame metrics and
    Idle segments from segment metrics. Unlabelled seconds (gt < 0) are dropped.

    >>> m = video_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 2)
    >>> m["acc"], m["F1@50"], m["edit"], m["seg_ratio"], m["boundary_F1"]
    (1.0, 100.0, 100.0, 1.0, 1.0)
    >>> round(video_metrics(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), 2)["seg_ratio"], 2)
    2.0
    """
    pred, gt = np.asarray(pred), np.asarray(gt)
    keep = gt >= 0
    pred, gt = pred[keep], gt[keep]
    ign = (0,) if ignore_idle else ()
    fm = gt != 0 if ignore_idle else np.ones(len(gt), bool)
    pf, gf = pred[fm], gt[fm]
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(conf, (gf, pf), 1)
    tp = np.diag(conf).astype(float)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    present = (conf.sum(0) + conf.sum(1)) > 0
    if ignore_idle:
        present[0] = False
    f1c = np.where(present, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    jac = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
    out = {"acc": float((pf == gf).mean()) if len(gf) else float("nan"),
           "macro_F1": float(np.nanmean(f1c)), "jaccard": float(np.nanmean(jac))}
    seg_per = {}
    for k in overlaps:
        a, b, c, per = _seg_counts(pred, gt, k / 100, ign)
        out[f"F1@{k}"] = 100 * _f1(a, b, c)
        out[f"_seg{k}"] = (a, b, c)
        seg_per[k] = per
    out["edit"] = _edit(pred, gt, ign)
    bt = _boundary(pred, gt, tol)
    out["boundary_F1"] = _f1(*bt)
    out["_bnd"] = bt
    npred = sum(1 for c, _, _ in runs(pred) if c not in ign)
    ngt = sum(1 for c, _, _ in runs(gt) if c not in ign)
    out["seg_ratio"] = npred / max(1, ngt)
    out["_nseg"] = (npred, ngt)
    out["_conf"] = conf
    out["_seg_per50"] = seg_per[50] if 50 in seg_per else {}
    return out


PUBLIC = ("acc", "macro_F1", "jaccard", "F1@10", "F1@25", "F1@50", "edit", "boundary_F1", "seg_ratio")


def aggregate(per_video, n_classes):
    """
    Mean over videos of the per-video metrics, plus pooled versions (counts summed over videos,
    MS-TCN convention): ``pooled_F1@k``, ``pooled_acc``, ``pooled_macro_F1``, ``pooled_boundary_F1``,
    ``pooled_seg_ratio``; and per class: frame F1 / Jaccard and segmental F1@50 (pooled).
    """
    out = {k: float(np.mean([m[k] for m in per_video])) for k in PUBLIC}
    for k in (10, 25, 50):
        a, b, c = np.sum([m[f"_seg{k}"] for m in per_video], 0)
        out[f"pooled_F1@{k}"] = 100 * _f1(a, b, c)
    conf = np.sum([m["_conf"] for m in per_video], 0)
    tp = np.diag(conf).astype(float)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    sup = conf.sum(1) > 0
    f1c = np.where(sup, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    out["pooled_acc"] = float(tp.sum() / max(1, conf.sum()))
    out["pooled_macro_F1"] = float(np.nanmean(f1c))
    out["pooled_boundary_F1"] = _f1(*np.sum([m["_bnd"] for m in per_video], 0))
    npred, ngt = np.sum([m["_nseg"] for m in per_video], 0)
    out["pooled_seg_ratio"] = float(npred / max(1, ngt))
    per50 = np.zeros((n_classes, 3))
    for m in per_video:
        for c, v in m["_seg_per50"].items():
            per50[c] += v
    out["per_class"] = {
        "frame_F1": [None if np.isnan(v) else float(v) for v in f1c],
        "jaccard": [None if not s else float(tp[c] / max(1, tp[c] + fp[c] + fn[c])) for c, s in enumerate(sup)],
        "seg_F1@50": [None if per50[c].sum() == 0 else 100 * _f1(*per50[c]) for c in range(n_classes)],
        "support_s": conf.sum(1).tolist(),
    }
    out["confusion"] = conf.tolist()
    out["n_videos"] = len(per_video)
    return out


# ---- CLI ---------------------------------------------------------------------------
def parse_param(s):
    """
    ``key=value`` with int / float / bool / str values; comma-separated numbers give a list
    (per-phase values, e.g. ``margin=0.1,0.2,...``).

    >>> parse_param("margin=0.1,0.2"), parse_param("bin=5"), parse_param("mode=order")
    (('margin', [0.1, 0.2]), ('bin', 5), ('mode', 'order'))
    """
    k, v = s.split("=", 1)
    if "," in v:
        return k, [float(x) for x in v.split(",")]
    for cast in (int, float):
        try:
            return k, cast(v)
        except ValueError:
            pass
    return k, {"true": True, "false": False}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--post-dir", required=True, help="phase_posteriors directory")
    ap.add_argument("--decoder", default="hsmm", choices=sorted(DECODERS))
    ap.add_argument("--param", nargs="*", default=[], help="decoder parameters key=value")
    ap.add_argument("--prior-dir", help="fit priors on every video of this directory instead of leave-fold-out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = dict(parse_param(p) for p in args.param)
    videos, names = load_posteriors(args.post_dir)
    C = len(names)
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    fixed = None
    if args.prior_dir:
        pv, _ = load_posteriors(args.prior_dir)
        fixed = fit_priors([v["gt"] for v in pv.values()], C)
    cache, per = {}, {"all": [], "noidle": []}
    for v, d in videos.items():
        pri = fixed
        if pri is None:
            if d["fold"] not in cache:
                cache[d["fold"]] = fit_priors([u["gt"] for u in videos.values() if u["fold"] != d["fold"]], C)
            pri = cache[d["fold"]]
        lab = decode(args.decoder, d["probs"], pri, **params)
        np.save(out / "labels" / f"{v}.npy", lab.astype(np.int16))
        per["all"].append(video_metrics(lab, d["gt"], C))
        per["noidle"].append(video_metrics(lab, d["gt"], C, ignore_idle=True))
    res = {"decoder": args.decoder, "params": params, "classes": names,
           **{k: aggregate(v, C) for k, v in per.items()}}
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    m = res["all"]
    print(f"{args.decoder} {params}: F1@50 {m['F1@50']:.1f} edit {m['edit']:.1f} acc {m['acc']:.3f} "
          f"macroF1 {m['macro_F1']:.3f} bF1@5 {m['boundary_F1']:.3f} ratio {m['seg_ratio']:.2f}")


if __name__ == "__main__":
    main()
