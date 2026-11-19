"""
EXP-21 (focus on 8 target events): multi-task clip targets on the 4 Hz grid.

Per video, from the revised 4 Hz label CSV (``<label_dir>/RARPxx/RARPxx_all_labels.csv``, one row per
0.25 s, ``frame_id`` = row of ``frames_4fps.npy``) and the TAPIS teacher json
(``action_labels/RARPxx_actions.json``):

* events: 9 classes = background + the 8 target events. Each target event (fractional ``time_s``) gives
  a gaussian bump of width ``sigma_s`` on the grid; background = 1 - sum of the bumps (clipped at 0),
  then rows are renormalised. The other 25 events are not classes: rows within ``nontarget_margin_s``
  of one are masked from the event loss (and kept for phases and teacher). Rows with ``ignore=1`` are
  masked too and never used as clip centres.
* phases: 15 classes, the local order of the EXP-19 S phase head (``PHASE_IDS``).
* teacher: per second, ``act`` = max over regions of p(action) (14) and ``ins`` = max over regions of
  m2f_score * p_TAPIS(instrument) (7), as ``analisi/.../orsi_signals.py``; seconds with no region (or not
  covered) are NaN. A clip target is the max over the seconds its 16 frames cover; a clip whose seconds
  are all NaN is masked.

Nothing here prints labels; callers only log aggregates.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

SHORT = [
    "Grasping_catheter_tip",
    "Clip_or_coagulate_vas_deferens_-_right",
    "Identification_and_clipping_of_SV_arteries_-_right",
    "Metal_clip_on_right_pedicle",
]
LONG = ["Start_dissection", "Incision_of_Denonvilliers_fascia", "Transection_of_the_urethra", "Stitch_in_bladder"]
TARGETS = SHORT + LONG
EVENT_CLASSES = ["Idle"] + TARGETS  # index 0 = background
# global phase ids in the local order of the 15-class phase head of EXP-19 S (train_phases_distribution.csv)
PHASE_IDS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ACTIONS = ["Cauterize", "Close", "Cut", "Grasp", "Hold", "Open", "Open Something", "Pull", "Push", "Release",
           "Still", "Suction", "Travel", "Other"]
INSTRUMENTS = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
               "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]

# same explicit collapses as Orsi.collapse_event_dfs; left/right variants share the base name
_COLLAPSE = {
    "instrument_swap:_removal": "removal_of_robotic_instruments",
    "hemolock_clip_on_right_pedicle": "metal_clip_on_right_pedicle",
    "hemolock_clip_on_left_pedicle": "metal_clip_on_left_pedicle",
    "insert_gauze": "insert_hemostatic_agens",
    "cutting_the_needles": "removing_needles",
}


def _norm(name):
    return str(name).strip().lower().replace(" ", "_")


def _base(name):
    n = _COLLAPSE.get(_norm(name), _norm(name))
    n = n.replace("_left", "").replace("_right", "")
    return n.replace("left_", "").replace("right_", "")


_TARGET_BASE = {_base(t): k + 1 for k, t in enumerate(TARGETS)}


def target_index(event_name):
    """
    1..8 for a target event (after the Orsi collapses), 0 for any other event, -1 for Idle.

    >>> target_index("Hemolock clip on left pedicle"), target_index("Clip or coagulate vas deferens - left")
    (4, 2)
    >>> target_index("Idle"), target_index("Start dissection DVC"), target_index("Stitch in bladder")
    (-1, 0, 8)
    """
    if _norm(name := event_name) == "idle":
        return -1
    return _TARGET_BASE.get(_base(name), 0)


class VideoLabels:
    """Grid labels of one video: ``n`` rows, ``events`` [(time_s, target_index)], ``phase``, ``ignore``."""

    def __init__(self, csv_path, fps=4):
        import pandas as pd

        df = pd.read_csv(csv_path)
        grid = df.drop_duplicates("frame_id").sort_values("frame_id")
        assert (grid["frame_id"].to_numpy() == np.arange(len(grid))).all(), csv_path
        self.n = len(grid)
        self.fps = fps
        ph = {g: i for i, g in enumerate(PHASE_IDS)}
        self.phase = np.array([ph.get(int(p), 0) for p in grid["phase_id"]], dtype=np.int64)
        ign = df.groupby("frame_id")["ignore"].max() if "ignore" in df else None
        self.ignore = (ign.reindex(range(self.n)).fillna(0).to_numpy() > 0) if ign is not None else np.zeros(self.n, bool)
        ev = df[df["event_name"].astype(str).str.strip().str.lower() != "idle"]
        tcol = "time_s" if "time_s" in ev else "second"
        self.events = [(float(t), target_index(nm)) for t, nm in zip(ev[tcol], ev["event_name"])]


def event_targets(n, events, fps=4, sigma_s=0.5, nontarget_margin_s=1.0, ignore=None, min_bump=1e-3):
    """
    Soft event targets [n, 9] (rows sum to 1) and the event-loss mask [n] (True = used).

    >>> y, m = event_targets(40, [(5.0, 3), (8.0, 0)], fps=4, sigma_s=0.5, nontarget_margin_s=1.0)
    >>> int(y[20].argmax()), round(float(y[20, 3]), 3), bool(m[20]), bool(m[32]), bool(m[27]), bool(m[28])
    (3, 1.0, True, False, True, False)
    """
    t = np.arange(n) / fps
    bumps = np.zeros((n, len(TARGETS)), np.float32)
    mask = np.ones(n, bool)
    for te, k in events:
        if k > 0:
            lo, hi = max(0, int(math.floor((te - 5 * sigma_s) * fps))), min(n, int(math.ceil((te + 5 * sigma_s) * fps)) + 1)
            g = np.exp(-0.5 * ((t[lo:hi] - te) / sigma_s) ** 2)
            g[g < min_bump] = 0
            bumps[lo:hi, k - 1] = np.maximum(bumps[lo:hi, k - 1], g)
        elif k == 0:
            mask &= ~(np.abs(t - te) <= nontarget_margin_s + 1e-9)
    y = np.concatenate([np.clip(1 - bumps.sum(1, keepdims=True), 0, None), bumps], 1)
    y /= y.sum(1, keepdims=True)
    # a target bump wins over a nearby non-target mask (the row is a positive of a target)
    mask |= bumps.max(1) >= 0.5
    if ignore is not None:
        mask &= ~np.asarray(ignore, bool)
    return y.astype(np.float32), mask


def teacher_seconds(action_json, n_seconds=None):
    """Per-second teacher signals: (act [S, 14], ins [S, 7]), NaN where a second has no region."""
    d = json.load(open(action_json))
    assert d["action_classes"] == ACTIONS and d["instrument_classes"] == INSTRUMENTS, action_json
    secs = d["seconds"]
    n = n_seconds or (max(s["second"] for s in secs) + 1)
    act = np.full((n, len(ACTIONS)), np.nan, np.float32)
    ins = np.full((n, len(INSTRUMENTS)), np.nan, np.float32)
    for row in secs:
        s = row["second"]
        if not 0 <= s < n or not row["regions"]:
            continue
        regs = row["regions"]
        act[s] = np.array([[r["actions"][a] for a in ACTIONS] for r in regs]).max(0)
        sc = np.array([r["m2f_score"] for r in regs])[:, None]
        ins[s] = (sc * np.array([[r["tapis_instruments"][i] for i in INSTRUMENTS] for r in regs])).max(0)
    return act, ins


def teacher_cache(patient, action_dir, cache_dir):
    """Cached ``teacher_seconds`` of one video (npz per video in ``cache_dir``)."""
    path = Path(cache_dir) / f"{patient}_teacher.npz"
    if not path.exists():
        act, ins = teacher_seconds(Path(action_dir) / f"{patient}_actions.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, act=act, ins=ins)
        os.replace(tmp, path)
    z = np.load(path)
    return z["act"], z["ins"]


def clip_rows(center, n, num_frames=16, sample_rate=1):
    """Grid rows of the clip centred at ``center`` (same rule as utils.get_sequence, mode center)."""
    lo = math.floor(center - num_frames * sample_rate / 2)
    return np.clip(np.arange(lo, lo + num_frames * sample_rate, sample_rate), 0, n - 1)


def teacher_window(sig, rows, fps=4):
    """
    Max over the seconds covered by the grid ``rows`` of a per-second signal [S, K] (NaN = masked).
    Returns (target [K], valid bool).

    >>> s = np.array([[0.1, 0.0], [np.nan, np.nan], [0.7, 0.2], [0.3, 0.9]], np.float32)
    >>> t, ok = teacher_window(s, np.arange(4, 12), fps=4)   # seconds 1 and 2
    >>> [round(float(v), 2) for v in t], ok
    ([0.7, 0.2], True)
    >>> teacher_window(s, np.arange(4, 8), fps=4)[1]          # only the masked second 1
    False
    """
    secs = np.unique(np.asarray(rows) // fps)
    secs = secs[secs < len(sig)]
    w = sig[secs]
    w = w[np.isfinite(w).all(1)]
    if len(w) == 0:
        return np.zeros(sig.shape[1], np.float32), False
    return w.max(0).astype(np.float32), True


def row_categories(y, mask, ignore, pos_thr=0.05):
    """
    Sampling category per row: 1..8 near a target (bump >= pos_thr, argmax class), 0 background,
    -1 masked near a non-target event, -2 never sampled (ignore).
    """
    cat = np.where(mask, 0, -1)
    bump = y[:, 1:]
    near = bump.max(1) >= pos_thr
    cat[near] = bump[near].argmax(1) + 1
    cat[np.asarray(ignore, bool)] = -2
    return cat


def sampling_weights(cats, share_target=0.5, share_background=0.4, share_nontarget=0.1):
    """
    Row weights (sum 1) over the concatenated ``cats`` of all training videos: ``share_target`` split
    equally among the target classes present, uniform inside a class; ``share_background`` uniform over
    background rows; ``share_nontarget`` over rows masked near non-target events; ignore rows get 0.
    """
    cats = np.asarray(cats)
    w = np.zeros(len(cats), np.float64)
    present = [k for k in range(1, len(TARGETS) + 1) if (cats == k).any()]
    for k in present:
        sel = cats == k
        w[sel] = share_target / len(present) / sel.sum()
    for code, share in ((0, share_background), (-1, share_nontarget)):
        sel = cats == code
        if sel.any():
            w[sel] = share / sel.sum()
    return w / w.sum()


def asymmetric_bce(p, y, valid, pos_weight=None, neg_weight=None, eps=1e-6):
    """
    Mean BCE of probabilities ``p`` [B, K] against soft targets ``y`` with per-class positive/negative
    weights ([K] tensors, default 1), over the rows where ``valid`` [B] is True. Torch tensors.
    """
    import torch

    p = p.float().clamp(eps, 1 - eps)
    pw = torch.ones(p.shape[1], device=p.device) if pos_weight is None else pos_weight
    nw = torch.ones(p.shape[1], device=p.device) if neg_weight is None else neg_weight
    loss = -(pw * y * torch.log(p) + nw * (1 - y) * torch.log(1 - p))
    v = valid.float()[:, None]
    return (loss * v).sum() / (v.sum() * p.shape[1]).clamp(min=1)


def soft_ce(logits, y, mask, class_weight=None):
    """Mean soft-target cross-entropy over rows where ``mask`` is True, per-class weights on the targets."""
    import torch

    logp = torch.log_softmax(logits.float(), 1)
    w = y if class_weight is None else y * class_weight
    loss = -(w * logp).sum(1)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1)
