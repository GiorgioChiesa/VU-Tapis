#!/usr/bin/env python3
"""
Hybrid model: frozen per-second clip features + an interchangeable global-context module.

Same data, targets, losses, folds and outputs as `tools/train_temporal.py` (imported,
not copied); the temporal model is any context module of
`tapis/models/temporal_context.py` (G0..G6) followed by the phase and event heads:

  G0 per-second MLP | G1 local TCN (1..32) | G2 MS-TCN (= train_temporal) | G3 (Bi)GRU
  G4 transformer on 8x tokens | G5 G1 + G4 | G6 LSTR-style memory (causal only)

With the same seed, `--context G2` consumes the random streams in the same order as
`train_temporal.py`, so `cv_grasp_dilate3` is reproduced run by run.

Outputs, in --out:
  model.pt, train_log.json
  val/<patient>.npz + val/meta.json   readable by eval_untrimmed.py / eval_constraints.py
  val/hybrid_metrics.json             quick spotting/phase metrics, segmental phase metrics
                                      (edit, F1@10/25/50), parameters, train time, latency

Inputs (--streams): `dynamic` clip features (--feat-dir), `static` center-frame features
(--static-feat-dir, `infer_untrimmed.py --static`), or `two` = both concatenated.

Datasets (--datasets, one at a time): `orsi` (CV folds as above) and `grasp` (official
train 8 / test 5 split from `tools/prepare_external.py`; phase head = 11 GraSP phases,
event head = onsets of the 20 steps and 10 phases, the test videos are written as `val`).

Experiment 2 (Orsi only): what phases and atomic actions add to event recognition.
  --phase-weight 0     events only, no phase supervision (arm 2.0)
  --phase-cond         event logits += log(softmax(phase) @ sigmoid(W)), W [P, E] learned,
                       initialised from the train-fold event/phase co-occurrence (arm 2.2)
  --oracle-phase       one-hot ground-truth phase appended to the input (upper bound, arm 2.3)
  --aux-mode stack --aux-sources S...
                       per-second posteriors of the auxiliary predictors of `tools/train_aux.py`
                       (trained on GraSP / SAR-RARP50 / ESAD, never on Orsi) appended to the input
  --aux-mode joint --aux-sources S...
                       one shared trunk, extra output channels per auxiliary task, trained on the
                       Orsi train fold plus every video of the source datasets (masked losses)
  sources: grasp_phase (GraSP phases + steps), grasp_action (GraSP actions + instruments),
           sar (SAR-RARP50 suturing gestures), esad (ESAD actions); see train_aux.SOURCES.

Experiment 3 (Orsi only): a phase-change (boundary) head shared by phases and events. Every Orsi
phase START is an event stamp, so for the "boundary" event classes (>= 50 % of the train-fold
onsets within 2 s of a phase change) spotting the event and segmenting the phases are one problem.
  --boundary-head      one more output channel: phase change at t (same target and focal loss
                       as the events, weight --boundary-weight); val npz get `probs_boundary`
  --boundary-fuse add  boundary-class event logits += a_c * boundary logit (a_c learned, init 0)
  --boundary-fuse transition
                       add, plus b_c * log P(phase the class opens, at t + 2 s) (b_c learned, init 0)
  --oracle-boundary    ground-truth phase changes (dilated +/- k s, no phase identity) appended to
                       the input: upper bound of knowing *when* the phase changes

Experiment 4 (Orsi only): T-DEED-style displacement head for fine localisation.
  --offset-head agnostic|class
                       1 (any class) or C (per class) channels regressing onset - t (s) on the seconds
                       within --offset-radius (default k) of an onset (SmoothL1, --offset-weight);
                       at inference every event score moves to t + offset (max) before the NMS. Val npz:
                       `probs_steps` displaced, `probs_steps_raw` and `offsets` for analysis.

Example:
    python tools/train_hybrid.py --context G4 --causal --target dilate --k 3 --cv-fold 0/5 \
        --feat-dir outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all \
        --out outputs/orsi/HYBRID/G4c_dilate3/seed0/fold0 --gpu 2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_aux
from eval_constraints import read_phase_labels
from eval_untrimmed import build_label_table, global_to_local, idle_index, load_cfg
from train_temporal import OFFSET_EVENTS, event_targets, losses, predict, quick_eval
from tapis.evaluate import segment_eval
from tapis.evaluate import spotting_eval as se
from tapis.models.temporal_context import CONTEXTS, build_context, count_parameters


# ---- data --------------------------------------------------------------------------
def cv_split(patients, fold):
    """Video-level fold ``i/n`` (same shuffle as train_temporal.py)."""
    i, n = map(int, fold.split("/"))
    shuffled = sorted(patients)
    random.Random(1234).shuffle(shuffled)
    held_out = set(shuffled[i::n])
    return {
        "train": [p for p in patients if p not in held_out],
        "val": [p for p in patients if p in held_out],
        "test": [],
    }


def load_orsi(args, out):
    """
    Orsi adapter: per-patient dict ``{x, phase, y, m, second, events}`` plus the split and
    the class vocabularies. Mirrors train_temporal.main (same tables, same order).
    """
    feat_dir = Path(args.feat_dir).resolve()
    fmeta = json.loads((feat_dir / "meta.json").read_text())
    run = Path(fmeta["run"])
    cfg = load_cfg(run, fmeta["ckpt"])
    split = {
        s: [Path(f).name.removesuffix(".csv") for f in getattr(cfg.ENDOVIS_DATASET, f"{s.upper()}_LISTS")]
        for s in ("train", "val", "test")
    }
    patients = split["train"] + split["val"] + split["test"]
    if args.cv_fold:
        split = cv_split(patients, args.cv_fold)

    table, ids = build_label_table(cfg, patients, args.frame_dir, out)
    g2l, names = global_to_local(run, "steps", ids["steps"])
    g2l_ph, ph_names = global_to_local(run, "phases", ids["phases"])
    offset = {i for i, n in enumerate(names) if n.lower() in OFFSET_EVENTS}
    events = table[(table["event_name"].str.strip().str.lower() != "idle") & table["event_id"].isin(g2l)]

    data = {}
    for p in patients:
        npz = np.load(feat_dir / f"{p}.npz")
        x = load_features(args, npz, p)
        ph = read_phase_labels(args.frame_dir, p, g2l_ph)
        ev = [(g2l[r.event_id], float(r.second)) for r in events[events["patient_name"] == p].itertuples()]
        y, m = event_targets(len(x), ev, len(names), args.target, args.k, args.sigma, offset)
        data[p] = {"x": x, "phase": ph, "y": y, "m": m, "second": npz["second"], "events": ev}
    info = {"feat_dir": feat_dir, "fmeta": fmeta, "run": run, "names": names, "ph_names": ph_names}
    return data, split, info


def onset_events(labels, first_class):
    """
    ``(class, second)`` at the first second of every non-Idle (label > 0) segment; label ``c``
    becomes class ``first_class + c - 1``.

    >>> onset_events([0, 2, 2, 0, 1, 1, 2], first_class=1)
    [(2, 1.0), (1, 4.0), (2, 6.0)]
    """
    labels = np.asarray(labels)
    start = np.r_[True, labels[1:] != labels[:-1]]
    return [(first_class + int(labels[t]) - 1, float(t)) for t in np.flatnonzero(start & (labels > 0))]


def load_grasp(args, out, label_dir="/home/gchie/workspace/data/external_1fps/grasp"):
    """
    GraSP adapter (labels of `tools/prepare_external.py`): dense phases for the phase head,
    onsets of steps and phases as point events. No CV: train -> test (written as "val").
    """
    if args.cv_fold:
        raise ValueError("GraSP uses its official train/test split, not --cv-fold")
    label_dir = Path(label_dir)
    lmeta = json.loads((label_dir / "meta.json").read_text())
    feat_dir = Path(args.feat_dir).resolve()
    fmeta = json.loads((feat_dir / "meta.json").read_text())
    steps, phases = lmeta["vocab"]["steps"], lmeta["vocab"]["phases"]
    assert steps[0].lower() == "idle" and phases[0].lower() == "idle"
    names = ["Idle"] + [f"step:{n}" for n in steps[1:]] + [f"phase:{n}" for n in phases[1:]]
    split = {"train": lmeta["splits"]["train"], "val": lmeta["splits"]["test"], "test": []}
    data = {}
    for p in split["train"] + split["val"]:
        lab = np.load(label_dir / p / "labels.npz")
        npz = np.load(feat_dir / f"{p}.npz")
        if not np.array_equal(lab["second"], npz["second"]):
            raise RuntimeError(f"{p}: labels and features are not aligned")
        x = load_features(args, npz, p)
        ev = onset_events(lab["steps"], 1) + onset_events(lab["phases"], len(steps))
        y, m = event_targets(len(x), ev, len(names), args.target, args.k, args.sigma)
        data[p] = {"x": x, "phase": lab["phases"].astype(np.int64), "y": y, "m": m, "second": npz["second"],
                   "events": ev}
    info = {"feat_dir": feat_dir, "fmeta": dict(fmeta, tasks=["phases", "events"]), "run": Path(fmeta["run"]),
            "names": names, "ph_names": phases}
    return data, split, info


def load_features(args, npz, p):
    """Input of video ``p`` (--streams): dynamic clip features, static ones, or both concatenated."""
    if args.streams == "dynamic":
        return npz["features"].astype(np.float32)
    st = np.load(Path(args.static_feat_dir) / f"{p}.npz")
    if not np.array_equal(st["second"], npz["second"]):
        raise RuntimeError(f"{p}: static and dynamic features are not aligned")
    x = st["features"] if args.streams == "static" else np.concatenate([npz["features"], st["features"]], 1)
    return x.astype(np.float32)


DATASETS = {"orsi": load_orsi, "grasp": load_grasp}


# ---- Experiment 2: extra inputs and auxiliary datasets ------------------------------
def aux_tasks(sources):
    return [t for s in sources for t in train_aux.SOURCES[s]]


def augment_inputs(args, data, n_phases):
    """Append the one-hot true phase (--oracle-phase) and/or stacked auxiliary posteriors to ``x``."""
    for p, d in data.items():
        extra = []
        if args.oracle_phase:
            extra.append(np.eye(n_phases, dtype=np.float32)[d["phase"]])
        if getattr(args, "oracle_boundary", False):
            extra.append(boundary_targets(d["phase"], "dilate", args.k)[0])
        if args.aux_mode == "stack":
            for s in args.aux_sources:
                ds = "grasp" if s.startswith("grasp") else s
                npz = np.load(Path(args.aux_dir) / ds / f"{p}.npz")
                if not np.array_equal(npz["second"], d["second"]):
                    raise RuntimeError(f"{p}: auxiliary posteriors of {ds} are not aligned")
                extra += [npz[t].astype(np.float32) for t in train_aux.SOURCES[s]]
        if extra:
            d["x"] = np.concatenate([d["x"]] + extra, 1)


def load_aux_videos(args):
    """Joint mode: every video of the source datasets, ``{(dataset, video): (x, targets)}`` + the task groups."""
    tasks = aux_tasks(args.aux_sources)
    groups, videos = [], {}
    for ds in dict.fromkeys(("grasp" if s.startswith("grasp") else s) for s in args.aux_sources):
        sizes, meta = train_aux.vocab_sizes(ds)
        mine = [t for t in train_aux.tasks_of(ds) if t in tasks]
        groups += [(t, train_aux.TASKS[t][2], sizes[t]) for t in mine]
        for v in [v for vs in meta["splits"].values() for v in vs]:
            videos[(ds, v)] = train_aux.load_video(ds, v, mine)
    return videos, groups


def phase_event_prior(data, train, n_phases, n_events, floor=0.05):
    """
    ``[P, E]`` compatibility: share of the train-fold events of class ``e`` that fall in phase
    ``p``, normalised to 1 per event class, floored (classes never seen get a flat prior).

    >>> d = {"a": {"phase": np.array([0, 1, 1, 2]), "events": [(1, 1.0), (1, 2.0), (2, 3.0)]}}
    >>> phase_event_prior(d, ["a"], 3, 3, floor=0.05).round(2).T.tolist()
    [[1.0, 1.0, 1.0], [0.05, 1.0, 0.05], [0.05, 0.05, 1.0]]
    """
    counts = np.zeros((n_phases, n_events))
    for p in train:
        for c, t in data[p]["events"]:
            counts[data[p]["phase"][min(int(round(t)), len(data[p]["phase"]) - 1)], c] += 1
    prior = np.ones((n_phases, n_events))
    seen = counts.sum(0) > 0
    prior[:, seen] = np.maximum(counts[:, seen] / counts[:, seen].max(0, keepdims=True), floor)
    return prior


# ---- Experiment 3: phase-change (boundary) head ---------------------------------------
BOUNDARY_LEAD = 2  # s: the phase a boundary event opens is read this long after the onset


def boundary_targets(phase, kind, k=3, sigma=2.0):
    """Target ``y`` and mask ``m`` ([T, 1]) of the phase-change channel, built like the event targets.

    >>> y, m = boundary_targets(np.array([0, 0, 0, 1, 1, 1, 1]), "dilate", k=1)
    >>> y[:, 0].astype(int).tolist(), int(m.min())
    ([0, 0, 1, 1, 1, 0, 0], 1)
    """
    return event_targets(len(phase), [(0, float(t)) for t in segment_eval.changes(phase)], 1, kind, k, sigma)


def boundary_fraction(phase, onsets, tol):
    """Fraction of ``onsets`` within ``tol`` s of a change in the per-second ``phase`` sequence.

    >>> boundary_fraction(np.array([0, 0, 1, 1, 1, 2]), [2.0, 4.0], 0)
    0.5
    """
    ch = segment_eval.changes(phase)
    if not len(onsets) or not len(ch):
        return 0.0
    return float(np.mean([np.abs(ch - t).min() <= tol for t in onsets]))


def boundary_classes(data, train, n_events, tol=2.0, min_frac=0.5):
    """Event classes with >= ``min_frac`` of their train-fold onsets within ``tol`` s of a phase change.

    >>> d = {"a": {"phase": np.array([0, 0, 0, 1, 1, 1]), "events": [(1, 3.0), (1, 4.0), (1, 2.0), (2, 0.0)]}}
    >>> boundary_classes(d, ["a"], 3)
    [1]
    """
    near, tot = np.zeros(n_events), np.zeros(n_events)
    for p in train:
        by_class = {}
        for c, t in data[p]["events"]:
            by_class.setdefault(c, []).append(t)
        for c, on in by_class.items():
            near[c] += boundary_fraction(np.asarray(data[p]["phase"]), on, tol) * len(on)
            tot[c] += len(on)
    return [c for c in range(n_events) if tot[c] and near[c] / tot[c] >= min_frac]


def opened_phase(data, train, classes, lead=BOUNDARY_LEAD):
    """Phase most often in place ``lead`` s after the train-fold onsets of each class of ``classes``.

    >>> d = {"a": {"phase": np.array([0, 0, 0, 1, 1, 1]), "events": [(1, 3.0), (1, 2.0), (2, 0.0)]}}
    >>> opened_phase(d, ["a"], [1, 2])
    {1: 1, 2: 0}
    """
    votes = {c: {} for c in classes}
    for p in train:
        ph = data[p]["phase"]
        for c, t in data[p]["events"]:
            if c in votes:
                q = int(ph[min(int(round(t)) + lead, len(ph) - 1)])
                votes[c][q] = votes[c].get(q, 0) + 1
    return {c: max(v, key=v.get) for c, v in votes.items() if v}


def boundary_loss(logits, y, m, gamma):
    """Focal BCE of the phase-change channel (``[B, 1, T]`` per stage), same form as the event loss."""
    total = 0.0
    for b in logits:
        prob = torch.sigmoid(b)
        bce = F.binary_cross_entropy_with_logits(b, y, reduction="none")
        total = total + ((torch.abs(y - prob) ** gamma) * bce * m).sum() / torch.clamp(y.sum(), min=1.0)
    return total


# ---- Experiment 4: offset (displacement) head, T-DEED style ----------------------------
def offset_targets(length, events, n_classes, radius, per_class):
    """
    Displacement ``d = onset - t`` (s) to the nearest onset within ``radius`` s and its mask, ``[T, K]``:
    one column per class (``per_class``) or one column for the nearest onset of any class.

    >>> d, m = offset_targets(8, [(1, 3.0), (0, 6.0)], 2, 1, per_class=False)
    >>> d[:, 0].tolist(), m[:, 0].astype(int).tolist()
    ([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0], [0, 0, 1, 1, 1, 1, 1, 1])
    >>> d, m = offset_targets(8, [(1, 3.0), (0, 6.0)], 2, 1, per_class=True)
    >>> m.astype(int).T.tolist()
    [[0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0, 0]]
    """
    k_cols = n_classes if per_class else 1
    d = np.zeros((length, k_cols), dtype=np.float32)
    best = np.full((length, k_cols), np.inf, dtype=np.float32)
    t = np.arange(length)
    for c, t0 in events:
        t0, k = int(round(t0)), (c if per_class else 0)
        dist = np.abs(t - t0).astype(np.float32)
        upd = (dist <= radius) & (dist < best[:, k])
        d[upd, k] = (t0 - t)[upd]
        best[upd, k] = dist[upd]
    return d, np.isfinite(best).astype(np.float32)


def offset_loss(preds, d, m):
    """SmoothL1 of the predicted displacements (``[B, K, T]`` per stage) on the masked seconds."""
    total = 0.0
    for o in preds:
        total = total + (F.smooth_l1_loss(o, d, reduction="none", beta=1.0) * m).sum() / torch.clamp(m.sum(), min=1.0)
    return total


def displace_scores(probs, d, radius, fg, bg):
    """
    Move every per-second event score to ``t + d`` (clipped to +/- ``radius``, rounded) keeping the
    maximum, then recompute the background column; ``d`` is ``[T, 1]`` or ``[T, C]``.

    >>> p = np.zeros((6, 2)); p[2, 1] = 0.9; p[3, 1] = 0.8; p[:, 0] = 1 - p[:, 1]
    >>> d = np.zeros((6, 1)); d[2, 0] = 0.6; d[3, 0] = 0.2
    >>> out = displace_scores(p, d, 3, [1], [0])
    >>> out[:, 1].round(2).tolist(), out[:, 0].round(2).tolist()
    ([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], [1.0, 1.0, 1.0, 0.1, 1.0, 1.0])
    >>> bool(np.allclose(displace_scores(p, np.zeros((6, 1)), 3, [1], [0]), p))
    True
    """
    n = len(probs)
    out = np.zeros_like(probs)
    t = np.arange(n)
    for c in fg:
        dc = d[:, c] if d.shape[1] > 1 else d[:, 0]
        target = np.clip(np.rint(t + np.clip(dc, -radius, radius)), 0, n - 1).astype(int)
        col = np.zeros(n, dtype=probs.dtype)
        np.maximum.at(col, target, probs[:, c])
        out[:, c] = col
    if bg:
        out[:, list(bg)] = 1.0 - out[:, fg].max(1, keepdims=True)
    return out


def weighted_losses(outs, y_phase, y_event, m_event, fg, gamma, smooth_w, phase_w):
    """`train_temporal.losses` with the phase terms (CE + T-MSE) scaled by ``phase_w``."""
    if phase_w == 1.0:
        return losses(outs, y_phase, y_event, m_event, fg, gamma, smooth_w)
    total = 0.0
    for p, e in outs:
        if phase_w:
            logp = F.log_softmax(p, 1)
            tmse = torch.clamp((logp[:, :, 1:] - logp.detach()[:, :, :-1]) ** 2, max=16).mean()
            total = total + phase_w * (F.cross_entropy(p, y_phase) + smooth_w * tmse)
        e, y, m = e[:, fg], y_event[:, fg], m_event[:, fg]
        prob = torch.sigmoid(e)
        bce = F.binary_cross_entropy_with_logits(e, y, reduction="none")
        total = total + ((torch.abs(y - prob) ** gamma) * bce * m).sum() / torch.clamp(y.sum(), min=1.0)
    return total


# ---- model -------------------------------------------------------------------------
class HybridModel(nn.Module):
    """
    Context module + phase/event heads, with the calling convention of
    train_temporal.TemporalModel: ``x [B, D, T] -> [(phase [B, P, T], event [B, E, T]), ...]``
    (one pair per supervised stage, the last one is the prediction).

    Experiment 2: ``n_aux`` extra output channels for the auxiliary tasks (`--aux-mode joint`;
    `forward` drops them, `forward_full` returns ``(phase, event, aux)``) and ``phase_prior``
    (``[P, E]``, `--phase-cond`): event logits += log(softmax(phase) @ sigmoid(W)), W learned.

    Experiment 3: ``boundary`` adds one last output channel (phase change, after the auxiliary ones,
    returned inside ``aux``); ``fuse = (classes, opened)`` adds to the logits of the boundary event
    ``classes`` a_c * boundary logit and, when ``opened`` (phase opened by each class) is given,
    b_c * log P(opened phase at t + BOUNDARY_LEAD). a, b are learned, initialised at 0.

    Experiment 4: ``n_offset`` displacement channels (1 or one per class), after the auxiliary
    channels and before the boundary one; `offsets(aux)` extracts them.
    """

    def __init__(self, context, dim_in, n_phases, n_events, causal=False, n_aux=0, phase_prior=None, boundary=False,
                 fuse=None, n_offset=0, **kwargs):
        super().__init__()
        self.n_phases, self.n_events, self.n_aux, self.n_offset = n_phases, n_events, n_aux, n_offset
        self.context = build_context(context, dim_in, n_phases + n_events + n_aux + n_offset + int(boundary),
                                     causal=causal, n_phases=n_phases, **kwargs)
        self.phase_cond = None
        if phase_prior is not None:
            w = torch.as_tensor(phase_prior, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
            self.phase_cond = nn.Parameter(torch.log(w / (1 - w)))
        self.fuse_gate = self.fuse_trans = None
        if fuse is not None:
            classes, opened = fuse
            self.register_buffer("fuse_idx", torch.as_tensor(classes, dtype=torch.long))
            self.fuse_gate = nn.Parameter(torch.zeros(len(classes)))
            if opened is not None:
                self.register_buffer("fuse_phase", torch.as_tensor(opened, dtype=torch.long))
                self.fuse_trans = nn.Parameter(torch.zeros(len(classes)))

    def offsets(self, a):
        """Displacement channels ``[B, n_offset, T]`` of the extra outputs ``a``."""
        return a[:, self.n_aux : self.n_aux + self.n_offset]

    def fuse_boundary(self, p, e, b):
        """Event logits ``e`` [B, E, T] + the boundary terms on the fused classes."""
        add = self.fuse_gate[None, :, None] * b
        if self.fuse_trans is not None:
            logp = F.log_softmax(p, 1)[:, self.fuse_phase]
            lead = min(BOUNDARY_LEAD, logp.shape[2] - 1)
            if lead > 0:  # value at t + lead, last value repeated at the end
                logp = torch.cat([logp[:, :, lead:], logp[:, :, -1:].expand(-1, -1, lead)], 2)
            add = add + self.fuse_trans[None, :, None] * logp
        return e.index_add(1, self.fuse_idx, add)

    def forward_full(self, x):
        outs = []
        for o in self.context.forward_all(x.transpose(1, 2)):
            o = o.transpose(1, 2)
            p, e, a = o[:, : self.n_phases], o[:, self.n_phases : self.n_phases + self.n_events], o[:, self.n_phases + self.n_events :]
            if self.phase_cond is not None:
                compat = torch.einsum("bpt,pe->bet", F.softmax(p, 1), torch.sigmoid(self.phase_cond))
                e = e + torch.log(compat.clamp_min(1e-4))
            if self.fuse_gate is not None:
                e = self.fuse_boundary(p, e, a[:, -1:])
            outs.append((p, e, a))
        return outs

    def forward(self, x):
        return [(p, e) for p, e, _ in self.forward_full(x)]


@torch.no_grad()
def predict_offset(model, x, bg, fg, radius):
    """
    `predict` with the event scores moved by the predicted displacements (`displace_scores`):
    ``(displaced events, phases, raw events, displacements [T, K])``.
    """
    ev, ph = predict(model, x, bg, fg)
    d = model.offsets(model.forward_full(x)[-1][2])[0].T.cpu().numpy()
    return displace_scores(ev, d, radius, fg, bg), ph, ev, d.astype(np.float32)


@torch.no_grad()
def predict_boundary(model, x):
    """Per-second phase-change probability ``[T, 1]`` (last stage, last output channel)."""
    return torch.sigmoid(model.forward_full(x)[-1][2][:, -1:])[0].T.cpu().numpy().astype(np.float32)


@torch.no_grad()
def latency(model, x, device, repeats=3):
    """Seconds of GPU compute to process ``x`` (one full video), best of ``repeats``."""
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        model(x)
        torch.cuda.synchronize(device)
        best = min(best, time.perf_counter() - t)
    return best


# ---- main --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", required=True, choices=CONTEXTS)
    parser.add_argument("--feat-dir", required=True)
    parser.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", default="dilate", choices=["point", "dilate", "gauss", "asym"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feat-dropout", type=float, default=0.5)
    parser.add_argument("--crop", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--smooth-weight", type=float, default=0.15)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--cv-fold", help="i/n: video-level CV over all patients (GraSP features only)")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases (aggregate scalars only)")
    parser.add_argument("--wandb-project", default="OrsiTapis")
    parser.add_argument("--wandb-entity", default="eidos-giorgio")
    parser.add_argument("--wandb-group", help="default: hybrid/<config> (run dir without seed/fold)")
    # ---- Experiment 2 ----
    parser.add_argument("--datasets", nargs="+", default=["orsi"], choices=sorted(DATASETS),
                        help="training dataset (one at a time; auxiliary datasets go through --aux-*)")
    parser.add_argument("--aux-mode", default="none", choices=["none", "joint", "stack"])
    parser.add_argument("--aux-sources", nargs="*", default=[], choices=sorted(train_aux.SOURCES))
    parser.add_argument("--aux-dir", default="outputs/orsi/AUX/final",
                        help="stack: posteriors of `train_aux.py --mode final` (<aux-dir>/<dataset>/<patient>.npz)")
    parser.add_argument("--aux-ratio", type=float, default=1.0,
                        help="joint: auxiliary videos per Orsi train video in each epoch")
    parser.add_argument("--aux-weight", type=float, default=1.0, help="joint: weight of the auxiliary losses")
    parser.add_argument("--phase-weight", type=float, default=1.0, help="weight of the phase loss (0: events only)")
    parser.add_argument("--phase-cond", action="store_true", help="event head conditioned on the phase posterior")
    parser.add_argument("--oracle-phase", action="store_true", help="ground-truth phases as input (upper bound)")
    parser.add_argument("--streams", default="dynamic", choices=["dynamic", "static", "two"])
    parser.add_argument("--static-feat-dir", help="features of `infer_untrimmed.py --static` (--streams static|two)")
    # Experiment 3
    parser.add_argument("--boundary-head", action="store_true", help="phase-change output channel + its loss")
    parser.add_argument("--boundary-weight", type=float, default=1.0, help="weight of the phase-change loss")
    parser.add_argument("--boundary-fuse", default="none", choices=["none", "add", "transition"],
                        help="feed the phase-change channel into the boundary event classes (needs --boundary-head)")
    parser.add_argument("--oracle-boundary", action="store_true",
                        help="ground-truth phase changes (dilated +/- k s) as input (upper bound)")
    # Experiment 4
    parser.add_argument("--offset-head", default="none", choices=["none", "agnostic", "class"],
                        help="displacement head: one channel (nearest onset of any class) or one per class")
    parser.add_argument("--offset-weight", type=float, default=1.0, help="weight of the displacement loss")
    parser.add_argument("--offset-radius", type=int, help="seconds around an onset with a displacement target (default k)")
    args = parser.parse_args(argv)
    if len(args.datasets) > 1:
        parser.error("one --datasets at a time; use --aux-mode joint for multi-dataset training")
    exp2 = args.aux_mode != "none" or args.phase_cond or args.oracle_phase or args.phase_weight != 1.0
    exp34 = args.boundary_head or args.oracle_boundary or args.offset_head != "none"
    if (exp2 or exp34) and args.datasets != ["orsi"]:
        parser.error("Experiment 2-4 options apply to --datasets orsi")
    if args.offset_radius is None:
        args.offset_radius = args.k
    if args.boundary_fuse != "none" and not args.boundary_head:
        parser.error("--boundary-fuse needs --boundary-head")
    if (args.aux_mode == "none") != (not args.aux_sources):
        parser.error("--aux-mode joint|stack and --aux-sources go together")
    if args.streams != "dynamic" and not args.static_feat_dir:
        parser.error("--streams static|two needs --static-feat-dir")
    return args


def run_name(out, dataset):
    """
    ``.../<config>/seed<s>[/fold<i>]`` -> W&B run name and default group (prefixed by the dataset
    when it is not Orsi).

    >>> run_name(Path("x/HYBRID/G2_dilate3/seed0/fold1"), "orsi")
    ('hybrid/G2_dilate3/seed0/fold1', 'hybrid/G2_dilate3')
    >>> run_name(Path("x/GRASP/G2_dilate3/seed2"), "grasp")
    ('hybrid-grasp/G2_dilate3/seed2', 'hybrid-grasp/G2_dilate3')
    """
    parts = out.parts
    seed = [i for i, q in enumerate(parts) if q.startswith("seed")]
    parts = parts[seed[-1] - 1 :] if seed else parts[-1:]
    prefix = "hybrid" if dataset == "orsi" else f"hybrid-{dataset}"
    return f"{prefix}/{'/'.join(parts)}", f"{prefix}/{parts[0]}"


def subset_maps(preds, data, bg, names, deltas=(1, 2, 3, 4, 5)):
    """
    mAP per family of event classes named ``<family>:<name>`` (GraSP: step / phase onsets).
    """
    fams = sorted({n.split(":")[0] for n in names if ":" in n})
    dets = {p: se.decode_events(ev, 5.0, 0.0, background_classes=bg) for p, (ev, _) in preds.items()}
    gts = {p: data[p]["events"] for p in preds}
    out = {}
    for f in fams:
        other = tuple(i for i, n in enumerate(names) if not n.startswith(f + ":"))
        res = se.spotting_map(dets, gts, deltas, len(names), ignore_classes=other)
        out.update({f"onset_{f}_mAP@{d}s": res["map"][d] for d in (1, 5)})
        out[f"onset_{f}_avg_mAP_tight"] = se.average_map(res["map"], se.TIGHT_DELTAS)
    return out


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, split, info = DATASETS[args.datasets[0]](args, out)
    names, ph_names = info["names"], info["ph_names"]
    bg = idle_index(names)
    fg = [c for c in range(len(names)) if c not in bg]

    dim_feat = data[split["train"][0]]["x"].shape[1]
    augment_inputs(args, data, len(ph_names))
    aux_videos, aux_groups = load_aux_videos(args) if args.aux_mode == "joint" else ({}, [])
    n_aux = sum(n for _, _, n in aux_groups)

    mu = np.concatenate([data[p]["x"] for p in split["train"]]).mean(0)
    sd = np.concatenate([data[p]["x"] for p in split["train"]]).std(0) + 1e-6

    prior = phase_event_prior(data, split["train"], len(ph_names), len(names)) if args.phase_cond else None
    fuse, b_classes = None, []
    if args.boundary_head:
        for d in data.values():
            d["yb"], d["mb"] = boundary_targets(d["phase"], args.target, args.k, args.sigma)
        b_classes = [c for c in boundary_classes(data, split["train"], len(names)) if c in fg]
        if args.boundary_fuse != "none":
            opened = opened_phase(data, split["train"], b_classes)
            b_classes = [c for c in b_classes if c in opened]
            fuse = (b_classes, [opened[c] for c in b_classes] if args.boundary_fuse == "transition" else None)
        print(f"boundary classes (train fold): {[names[c] for c in b_classes]}", flush=True)
    n_offset = {"none": 0, "agnostic": 1, "class": len(names)}[args.offset_head]
    if n_offset:
        for d in data.values():
            d["yo"], d["mo"] = offset_targets(len(d["x"]), d["events"], len(names), args.offset_radius,
                                              per_class=args.offset_head == "class")
    model = HybridModel(args.context, len(mu), len(ph_names), len(names), args.causal, n_aux=n_aux, phase_prior=prior,
                        boundary=args.boundary_head, fuse=fuse, n_offset=n_offset, dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"{args.context}{' causal' if args.causal else ''}: {n_params / 1e6:.3f} M parameters", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def tensors(p, start=0, stop=None):
        d = data[p]
        sl = slice(start, stop)
        x = torch.from_numpy((d["x"][sl] - mu) / sd).T[None].to(device)
        return (
            x,
            torch.from_numpy(d["phase"][sl])[None].long().to(device),
            torch.from_numpy(d["y"][sl]).T[None].to(device),
            torch.from_numpy(d["m"][sl]).T[None].to(device),
        )

    def predict_events(p):
        """``(events, phases[, raw events, displacements])`` of video ``p``; displaced with --offset-head."""
        if n_offset:
            return predict_offset(model, tensors(p)[0], bg, fg, args.offset_radius)
        return predict(model, tensors(p)[0], bg, fg)

    def aux_step(key):
        """Loss of one auxiliary video (joint mode): trunk + its task channels only."""
        x, y = aux_videos[key]
        start = random.randint(0, max(0, len(x) - args.crop))
        sl = slice(start, start + args.crop)
        xt = torch.from_numpy((x[sl] - mu[:dim_feat]) / sd[:dim_feat]).T[None].to(device)
        if xt.shape[1] < len(mu):  # columns appended to the Orsi input (oracle phase) are absent here
            xt = torch.cat([xt, torch.zeros(1, len(mu) - xt.shape[1], xt.shape[2], device=device)], 1)
        xt = F.dropout1d(xt, args.feat_dropout, training=True) if args.feat_dropout else xt
        ys = {}
        for t, kind, _ in aux_groups:
            if t not in y:
                continue
            ys[t] = (torch.from_numpy(y[t][sl])[None].to(device) if kind == "softmax" else
                     (torch.from_numpy(y[t][0][sl]).T[None].to(device), torch.from_numpy(y[t][1][sl])[None].to(device)))
        mine = [g for g in aux_groups if g[0] in ys]
        offsets, i = {}, 0
        for t, _, n in aux_groups:
            offsets[t] = (i, i + n)
            i += n
        outs = [{t: a[:, offsets[t][0] : offsets[t][1]] for t, _, _ in mine} for _, _, a in model.forward_full(xt)]
        return train_aux.aux_loss(outs, ys, mine, args.smooth_weight)

    name, group = run_name(out, args.datasets[0])
    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            group=args.wandb_group or group,
            job_type="hybrid",
            dir=str(out),
            config={
                **vars(args),
                "features": str(info["feat_dir"]),
                "encoder_ckpt": info["fmeta"].get("encoder_ckpt") or str(info["run"] / info["fmeta"]["ckpt"]),
                "n_params": n_params,
                "n_train_videos": len(split["train"]),
                "n_val_videos": len(split["val"]),
            },
        )

    log = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(split["train"])
        if aux_videos:  # joint: Orsi videos + aux videos, sources sampled uniformly, then videos
            by_ds = {}
            for k in aux_videos:
                by_ds.setdefault(k[0], []).append(k)
            n_aux_videos = int(round(args.aux_ratio * len(order)))
            order += [random.choice(by_ds[random.choice(sorted(by_ds))]) for _ in range(n_aux_videos)]
        random.shuffle(order)
        ep_loss = 0.0
        for p in order:
            if isinstance(p, tuple):
                loss = args.aux_weight * aux_step(p)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                continue
            n = len(data[p]["x"])
            start = random.randint(0, max(0, n - args.crop))
            x, yp, ye, me = tensors(p, start, start + args.crop)
            x = F.dropout1d(x, args.feat_dropout, training=True) if args.feat_dropout else x
            if args.boundary_head or n_offset:
                full = model.forward_full(x)
                loss = weighted_losses([(a, b) for a, b, _ in full], yp, ye, me, fg, args.gamma, args.smooth_weight,
                                       args.phase_weight)
                crop = lambda k: torch.from_numpy(data[p][k][start : start + args.crop]).T[None].to(device)
                if args.boundary_head:
                    loss = loss + args.boundary_weight * boundary_loss([a[:, -1:] for _, _, a in full], crop("yb"),
                                                                       crop("mb"), args.gamma)
                if n_offset:
                    loss = loss + args.offset_weight * offset_loss([model.offsets(a) for _, _, a in full], crop("yo"),
                                                                   crop("mo"))
            else:
                loss = weighted_losses(model(x), yp, ye, me, fg, args.gamma, args.smooth_weight, args.phase_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        row = {"epoch": epoch + 1, "train/loss": ep_loss / len(split["train"]), "lr": sched.get_last_lr()[0]}
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                row["val/loss"] = float(
                    np.mean(
                        [losses(model(tensors(p)[0]), *tensors(p)[1:], fg, args.gamma, args.smooth_weight).item()
                         for p in split["val"]]
                    )
                )
            preds = {p: predict_events(p)[:2] for p in split["val"]}
            row.update({f"val/{k}": v for k, v in quick_eval(preds, data, bg, len(names)).items()})
            log.append(row)
            print(
                f"epoch {epoch + 1}: train {row['train/loss']:.4f} val {row['val/loss']:.4f} "
                f"mAP@5 {row['val/mAP@5s']:.3f} F1@5 {row['val/F1@5s']:.3f} ({time.time() - t_start:.0f}s)",
                flush=True,
            )
        if wb is not None:
            wb.log(row, step=epoch + 1)
    train_time = time.time() - t_start

    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "args": vars(args)}, out / "model.pt")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))

    model.eval()
    for s in ("val", "test"):
        pred_dir = out / s
        pred_dir.mkdir(exist_ok=True)
        if not split[s]:
            continue
        meta = dict(
            info["fmeta"], split=s, patients=split[s], train_patients=split["train"], temporal_model=str(out),
            target=args.target, cv_fold=args.cv_fold, context=args.context, causal=args.causal,
            streams=args.streams, dataset=args.datasets[0], event_names=names, aux_mode=args.aux_mode,
            aux_sources=args.aux_sources, phase_cond=args.phase_cond, oracle_phase=args.oracle_phase,
            phase_weight=args.phase_weight, boundary_head=args.boundary_head, boundary_fuse=args.boundary_fuse,
            oracle_boundary=args.oracle_boundary, boundary_classes=[names[c] for c in b_classes],
            offset_head=args.offset_head, offset_radius=args.offset_radius,
        )
        (pred_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        full_preds = {p: predict_events(p) for p in split[s]}
        preds = {p: v[:2] for p, v in full_preds.items()}
        for p, (ev, ph) in preds.items():
            extra = {"probs_boundary": predict_boundary(model, tensors(p)[0])} if args.boundary_head else {}
            if n_offset:
                extra.update(probs_steps_raw=full_preds[p][2].astype(np.float32), offsets=full_preds[p][3])
            np.savez_compressed(
                pred_dir / f"{p}.npz",
                second=data[p]["second"],
                probs_steps=ev.astype(np.float32),
                probs_phases=ph.astype(np.float32),
                **extra,
            )
        longest = max(split[s], key=lambda p: len(data[p]["x"]))
        lat = latency(model, tensors(longest)[0], device)
        metrics = {
            **quick_eval(preds, data, bg, len(names)),
            **subset_maps(preds, data, bg, names),
            **{f"phase_{k}": v for k, v in segment_eval.segmental_metrics(
                {p: ph.argmax(1) for p, (_, ph) in preds.items()}, {p: data[p]["phase"] for p in preds}
            ).items()},
            **{f"phase_boundary_{k}@{tol}s": v for tol in (2, 5, 10) for k, v in segment_eval.boundary_f1(
                {p: ph.argmax(1) for p, (_, ph) in preds.items()}, {p: data[p]["phase"] for p in preds}, tol
            ).items()},
            "n_params": n_params,
            "train_time_s": train_time,
            "latency_ms_per_video_hour": 1000 * lat / (len(data[longest]["x"]) / 3600),
        }
        (pred_dir / "hybrid_metrics.json").write_text(json.dumps(metrics, indent=2))
        if wb is not None:
            wb.summary.update({f"final_{s}/{k}": v for k, v in metrics.items()})
    if wb is not None:
        wb.finish()
    print(f"done: {out}")


if __name__ == "__main__":
    main()
