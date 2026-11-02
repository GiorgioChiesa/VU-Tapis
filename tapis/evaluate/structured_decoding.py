"""
Structured decoding of point events with priors estimated on the training videos.

Three kinds of constraints, all applied on top of per-second class probabilities
(no retraining):

* **phase gating** - an event is only plausible in the phases where it occurs in
  training: ``s(t, c) = p_event(t, c) * sum_phi p_phase(t, phi) * M[c, phi]`` with
  ``M[c, phi] = P(phi | c)`` rescaled so that its maximum over phases is 1;
* **cardinality** - "milestone" events occur a fixed small number of times per
  video (usually once): only the top-``cap`` detections of the class are kept;
* **order** - milestones that occur once follow a near-deterministic order: the
  subset of detections with strictly increasing times along that order and the
  largest total score is selected by dynamic programming (longest weighted path).

Detections that a constraint rejects are not dropped but multiplied by ``alpha``
(``alpha=0`` gives hard constraints), so recall is never capped by the decoder.
"""

from collections import defaultdict

import numpy as np


def estimate_priors(
    train_events,
    video_lengths,
    num_event_classes,
    num_phase_classes,
    background_classes=(0,),
    cap_max=2,
    cap_coverage=0.8,
    chain_min_presence=0.5,
    chain_min_consistency=0.9,
    phase_smoothing=1.0,
):
    """
    Args:
        train_events (dict): ``{video: [(event_class, t, phase_class), ...]}``.
        video_lengths (dict): ``{video: seconds}``.
        cap_max (int): largest per-video count treated as a cardinality constraint.
        cap_coverage (float): a class gets ``cap = mode`` of its per-video count if at
            least this fraction of the videos containing it have ``count <= mode``.
        chain_min_presence (float): a ``cap == 1`` class enters the order chain if it
            occurs in at least this fraction of training videos.
        chain_min_consistency (float): every ordered pair of the chain must agree with
            the chain order in at least this fraction of the videos containing both.
        phase_smoothing (float): additive smoothing of the event/phase counts.

    Returns:
        dict with ``phase_given_event`` [C_ev, C_ph] (max-normalised per row),
        ``caps`` {class: cap}, ``chain`` (ordered classes), ``precedence`` [C, C]
        (fraction of videos where the first occurrence of a precedes that of b) and
        ``median_rel_time`` per class.
    """
    videos = sorted(video_lengths)
    fg = [c for c in range(num_event_classes) if c not in background_classes]

    m = np.full((num_event_classes, num_phase_classes), phase_smoothing, dtype=np.float64)
    counts = np.zeros((num_event_classes, len(videos)), dtype=int)
    first = np.full((num_event_classes, len(videos)), np.nan)
    for j, v in enumerate(videos):
        for c, t, ph in train_events.get(v, []):
            m[c, ph] += 1
            counts[c, j] += 1
            first[c, j] = t if np.isnan(first[c, j]) else min(first[c, j], t)
    m /= m.max(axis=1, keepdims=True)

    caps = {}
    for c in fg:
        present = counts[c] > 0
        if not present.any():
            continue
        mode = int(np.bincount(counts[c][present]).argmax())
        if mode <= cap_max and (counts[c][present] <= mode).mean() >= cap_coverage:
            caps[c] = mode

    precedence = np.full((num_event_classes, num_event_classes), np.nan)
    for a in fg:
        for b in fg:
            both = ~np.isnan(first[a]) & ~np.isnan(first[b])
            if a != b and both.any():
                precedence[a, b] = float((first[a][both] < first[b][both]).mean())

    lengths = np.array([video_lengths[v] for v in videos], dtype=np.float64)
    rel = first / lengths[None, :]
    median_rel = {c: float(np.nanmedian(rel[c])) for c in fg if not np.all(np.isnan(rel[c]))}

    candidates = [
        c for c, cap in caps.items() if cap == 1 and (counts[c] > 0).mean() >= chain_min_presence
    ]
    chain = sorted(candidates, key=lambda c: median_rel[c])
    while True:
        bad = defaultdict(int)
        for i, a in enumerate(chain):
            for b in chain[i + 1 :]:
                if not precedence[a, b] >= chain_min_consistency:
                    bad[a] += 1
                    bad[b] += 1
        if not bad:
            break
        chain.remove(max(bad, key=lambda c: (bad[c], c)))

    return {
        "phase_given_event": m,
        "caps": caps,
        "chain": chain,
        "precedence": precedence,
        "median_rel_time": median_rel,
    }


def smooth(probs, window):
    """Centered moving average over time (``window`` seconds, odd), renormalised."""
    if window <= 1:
        return probs
    k = np.ones(int(window)) / int(window)
    out = np.stack([np.convolve(probs[:, i], k, mode="same") for i in range(probs.shape[1])], axis=1)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def phase_gate(event_probs, phase_probs, phase_given_event):
    """
    Multiply each event score by how compatible the (predicted) phase is with it.

    >>> ev = np.array([[0.2, 0.8], [0.2, 0.8]])
    >>> ph = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> M = np.array([[1.0, 1.0], [1.0, 0.1]])  # class 1 lives in phase 0
    >>> phase_gate(ev, ph, M).round(2).tolist()
    [[0.2, 0.8], [0.2, 0.08]]
    """
    gate = phase_probs @ phase_given_event.T  # [T, C_ev]
    return event_probs * gate


def apply_caps(detections, caps, alpha=0.0):
    """
    Keep the ``cap`` highest-scoring detections of each capped class in a video.

    >>> apply_caps([(1, 10, 0.9), (1, 50, 0.8), (2, 30, 0.6)], {1: 1}, alpha=0.5)
    [(1, 10, 0.9), (1, 50, 0.4), (2, 30, 0.6)]
    """
    by_class = defaultdict(list)
    for i, (c, _, s) in enumerate(detections):
        by_class[c].append((s, i))
    keep = set(range(len(detections)))
    for c, cap in caps.items():
        ranked = sorted(by_class.get(c, []), reverse=True)
        keep -= {i for _, i in ranked[cap:]}
    return [(c, t, s if i in keep else s * alpha) for i, (c, t, s) in enumerate(detections)]


def chain_decode(detections, chain, alpha=0.0, max_candidates=20):
    """
    Select at most one detection per chain class with strictly increasing times
    along ``chain`` and maximal total score; the other detections of chain classes
    are multiplied by ``alpha``. Classes outside the chain are left untouched.

    >>> dets = [(1, 50, 0.9), (2, 10, 0.8), (2, 60, 0.5), (3, 70, 0.6)]
    >>> chain_decode(dets, chain=[1, 2, 3])
    [(1, 50, 0.9), (2, 10, 0.0), (2, 60, 0.5), (3, 70, 0.6)]
    """
    rank = {c: k for k, c in enumerate(chain)}
    nodes = [
        (rank[c], t, s, i)
        for i, (c, t, s) in enumerate(detections)
        if c in rank
    ]
    # cap the number of candidates per class to keep the DP small
    per_class = defaultdict(list)
    for n in nodes:
        per_class[n[0]].append(n)
    nodes = [n for k in per_class for n in sorted(per_class[k], key=lambda x: -x[2])[:max_candidates]]
    nodes.sort(key=lambda n: (n[0], n[1]))

    best = np.zeros(len(nodes))
    back = np.full(len(nodes), -1)
    for j, (kj, tj, sj, _) in enumerate(nodes):
        best[j] = sj
        for i in range(j):
            ki, ti, _, _ = nodes[i]
            if ki < kj and ti < tj and best[i] + sj > best[j]:
                best[j] = best[i] + sj
                back[j] = i
    selected = set()
    if len(nodes):
        j = int(np.argmax(best))
        while j >= 0:
            selected.add(nodes[j][3])
            j = int(back[j])
    return [
        (c, t, s if (c not in rank or i in selected) else s * alpha)
        for i, (c, t, s) in enumerate(detections)
    ]
