"""
Online (causal) decoding of point events from per-second class scores.

A peak of class ``c`` at second ``t`` is *emitted* at second ``t + L`` (``L`` = confirmation
delay) if its score is

* strictly larger than every score in ``[t - w, t - 1]`` (``w`` = suppression window), and
* not smaller than every score in ``[t + 1, t + L]``,

so the decision at ``t + L`` only uses scores up to ``t + L``. At the end of the video the
peaks of the last ``L`` seconds are emitted at the last second with the truncated future.
A score threshold ``theta`` decides which emissions are alarms; every peak keeps its score
so that AP can still be computed on the ranked list.

Two causal constraints act on the emissions in emission order (same ``alpha`` semantics as
``structured_decoding``: a rejected emission keeps ``score * alpha``):

* **cap**: a per-class counter; once ``cap[c]`` alarms (score >= theta) of class ``c`` have
  been emitted, later emissions of ``c`` are multiplied by ``alpha``;
* **chain**: an order state machine over the milestone classes ``chain``.
  ``chain_rule="order"`` rejects an emission of a chain class when a *later* chain class has
  already been emitted as an alarm (the causal analogue of the offline longest-path
  decoding); ``chain_rule="predecessor"`` rejects it when one of its predecessors in the chain
  has not been emitted yet.

The same decoder is available in two forms that must give identical results:
:func:`decode_online` on a whole score matrix (vectorised peak search) and
:class:`StreamDecoder`, fed one second at a time.

Per-class parameters: ``theta``, ``window``, ``delay`` and ``cap`` may be scalars (global) or
dicts ``{class: value}`` (missing classes fall back to ``default``).
"""

from collections import defaultdict

import numpy as np


def _get(param, c, default=None):
    if isinstance(param, dict):
        return param.get(c, default)
    return param if param is not None else default


def online_peaks(scores, window, delay):
    """
    Peaks of one class with their emission second.

    Args:
        scores (np.ndarray): ``[T]`` scores of one class.
        window (int): past suppression window ``w`` (seconds).
        delay (int): confirmation delay ``L`` (seconds).

    Returns:
        list of ``(t, score, t_emit)`` sorted by ``t``; only positive scores.

    >>> online_peaks(np.array([0.1, 0.9, 0.8, 0.1, 0.0, 0.7]), window=2, delay=1)
    [(1, 0.9, 2), (5, 0.7, 5)]
    """
    s = np.asarray(scores, dtype=np.float64)
    T = len(s)
    if T == 0:
        return []
    w, L = int(window), int(delay)
    ninf = np.full(max(w, L, 1), -np.inf)
    if w > 0:
        past = np.lib.stride_tricks.sliding_window_view(np.concatenate([ninf[:w], s]), w)[:T].max(1)
    else:
        past = np.full(T, -np.inf)
    if L > 0:
        future = np.lib.stride_tricks.sliding_window_view(np.concatenate([s[1:], ninf[:L], [-np.inf]]), L)[:T].max(1)
    else:
        future = np.full(T, -np.inf)
    ok = (s > past) & (s >= future) & (s > 0)
    ts = np.flatnonzero(ok)
    return [(int(t), float(s[t]), int(min(t + L, T - 1))) for t in ts]


class CausalConstraints:
    """Cap / chain state machine applied to emissions in emission order."""

    def __init__(self, theta, caps=None, chain=(), alpha=0.1, chain_rule="order"):
        if chain_rule not in ("order", "predecessor"):
            raise ValueError(chain_rule)
        self.theta = theta
        self.caps = caps or {}
        self.rank = {c: k for k, c in enumerate(chain)}
        self.alpha = alpha
        self.chain_rule = chain_rule
        self.count = defaultdict(int)
        self.emitted_ranks = set()

    def __call__(self, c, t, s, t_emit):
        th = _get(self.theta, c, 0.0)
        cap = _get(self.caps, c)
        if cap is not None and self.count[c] >= cap:
            s *= self.alpha
        if c in self.rank:
            r = self.rank[c]
            if self.chain_rule == "order":
                violated = any(k > r for k in self.emitted_ranks)
            else:
                violated = any(k not in self.emitted_ranks for k in range(r))
            if violated:
                s *= self.alpha
        if s >= th and s > 0:
            self.count[c] += 1
            if c in self.rank:
                self.emitted_ranks.add(self.rank[c])
        return (c, t, s, t_emit)


def decode_online(probs, theta=0.0, window=5, delay=0, caps=None, chain=(), alpha=0.1,
                  chain_rule="order", background_classes=(0,), classes=None):
    """
    Online decoding of a whole video (vectorised peak search + causal constraints).

    Args:
        probs (np.ndarray): ``[T, C]`` per-second scores.
        theta, window, delay: scalars or ``{class: value}``.
        caps (dict): ``{class: cap}``; classes without a cap are not limited.
        chain (list): ordered milestone classes (empty = no chain).
        classes (iterable): classes to decode (default: all but background).

    Returns:
        list of ``(class, t, score, t_emit)`` in emission order.
    """
    probs = np.asarray(probs)
    classes = [c for c in range(probs.shape[1]) if c not in background_classes] if classes is None else classes
    em = []
    for c in classes:
        L = _get(delay, c, 0)
        for t, s, te in online_peaks(probs[:, c], _get(window, c, 5), L):
            # same order as StreamDecoder: by emission second, regular emissions before the
            # end-of-video flush, then by class and time
            em.append((te, t + L > te, c, t, s))
    em.sort()
    cons = CausalConstraints(theta, caps, chain, alpha, chain_rule)
    return [cons(c, t, s, te) for te, _, c, t, s in em]


class StreamDecoder:
    """
    The decoder of :func:`decode_online`, fed one second at a time.

    ``step(scores_t)`` returns the emissions decided at this second; ``flush()`` emits the
    peaks of the last seconds when the video ends.
    """

    def __init__(self, num_classes, theta=0.0, window=5, delay=0, caps=None, chain=(), alpha=0.1,
                 chain_rule="order", background_classes=(0,), classes=None):
        self.classes = (
            [c for c in range(num_classes) if c not in background_classes] if classes is None else list(classes)
        )
        self.window = {c: int(_get(window, c, 5)) for c in self.classes}
        self.delay = {c: int(_get(delay, c, 0)) for c in self.classes}
        self.cons = CausalConstraints(theta, caps, chain, alpha, chain_rule)
        self.hist = []  # scores seen so far (a ring buffer of max(w + L + 1) rows would do)

    def _is_peak(self, c, t, last):
        s = self.hist[t][c]
        if s <= 0:
            return False
        lo = max(0, t - self.window[c])
        if any(self.hist[u][c] >= s for u in range(lo, t)):
            return False
        return all(self.hist[u][c] <= s for u in range(t + 1, min(t + self.delay[c], last) + 1))

    def step(self, scores_t):
        self.hist.append(np.asarray(scores_t, dtype=np.float64))
        now = len(self.hist) - 1
        out = []
        for c in self.classes:
            t = now - self.delay[c]
            if t >= 0 and self._is_peak(c, t, now):
                out.append(self.cons(c, t, float(self.hist[t][c]), now))
        return out

    def flush(self):
        last = len(self.hist) - 1
        pending = []
        for c in self.classes:
            for t in range(max(0, last - self.delay[c] + 1), last + 1):
                if self._is_peak(c, t, last):
                    pending.append((c, t))
        return [self.cons(c, t, float(self.hist[t][c]), last) for c, t in sorted(pending)]
