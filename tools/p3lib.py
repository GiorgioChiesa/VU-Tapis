"""
EXP-21 phase 3 shared helpers: loaders for the encoder (arm M) and long-model predictions, fusion
primitives, event and phase evaluation with per-video values (for the paired tests), and the decoder
setup of tools/eval_focus8.py. Aggregates only; nothing per patient is printed.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "tools", REPO / "detectron2"):
    sys.path.insert(0, str(_p))

import eval_focus8 as ev  # noqa: E402
import phase_decode as pd_  # noqa: E402
from tapis.datasets import focus8 as F  # noqa: E402
from tapis.evaluate import spotting_eval as se  # noqa: E402

# inputs of phases 1-2 live in the focus8 worktree (read only, absolute paths)
FOCUS8 = Path("/home/gchie/workspace/wt/focus8")
RUN = FOCUS8 / "results/M/fold0"
ENC_DIR = RUN / "untrimmed"
LONG_DIR = RUN / "long"
SPLITS = RUN / "splits.json"
OUT = REPO / "results/p3"
LONG_NAMES = ["Idle"] + F.LONG
SHORT_IDX = [i for i, n in enumerate(F.EVENT_CLASSES) if n in F.SHORT]
LONG_IDX9 = [i for i, n in enumerate(F.EVENT_CLASSES) if n in F.LONG]
DELTAS = ev.DELTAS
LONG_RUNS = ["a3_s0", "a3_s1", "sed_s0", "sed_s1"]
CAUSAL_RUNS = ["a3_causal_s0", "a3_causal_s1", "sed_causal_s0", "sed_causal_s1"]


def splits():
    return json.loads(SPLITS.read_text())


def load_dir(d, videos):
    return {v: dict(np.load(Path(d) / f"{v}.npz")) for v in videos}


def load_long(run, st, videos):
    """Long-model predictions of one run name; ``a3_ens`` / ``sed_ens`` average the two seeds."""
    if run.endswith("_ens"):
        arch = run[:-4]
        zs = [load_dir(LONG_DIR / f"{arch}_s{s}" / st, videos) for s in (0, 1)]
        out = {}
        for v in videos:
            assert np.array_equal(zs[0][v]["second"], zs[1][v]["second"])
            out[v] = {"second": zs[0][v]["second"],
                      "probs_steps": (zs[0][v]["probs_steps"] + zs[1][v]["probs_steps"]) / 2,
                      "probs_phases": (zs[0][v]["probs_phases"] + zs[1][v]["probs_phases"]) / 2}
        return out
    return load_dir(LONG_DIR / run / st, videos)


def enc_long_view(z):
    """Encoder prediction restricted to the long events: columns [Idle, 4 long targets]."""
    p = z["probs_steps"].astype(np.float32)
    return {"second": z["second"], "probs_steps": np.concatenate([p[:, :1], p[:, LONG_IDX9]], 1),
            "probs_phases": z["probs_phases"].astype(np.float32)}


# ------------------------------------------------------------------------------------------ fusion
def confidence(p, events):
    """Per-step confidence: strongest event class (columns 1..) for events, top class for phases.

    >>> confidence(np.array([[0.7, 0.2, 0.1]]), True).tolist(), confidence(np.array([[0.7, 0.2, 0.1]]), False).tolist()
    ([0.2], [0.7])
    """
    return (p[:, 1:] if events else p).max(1)


def fuse_conf(pa, pb, ca, cb):
    """Per-step weighted mean, w = ca / (ca + cb) on a: equal confidences give the plain mean."""
    tot = ca + cb
    w = np.where(tot > 0, ca / np.where(tot > 0, tot, 1.0), 0.5)[:, None]
    return w * pa + (1.0 - w) * pb


def fuse_global(pa, pb, w):
    """w * a + (1 - w) * b (w = 0.5 is the plain mean)."""
    return w * pa + (1.0 - w) * pb


def fuse_videos(A, B, how, events, w=None):
    out = {}
    for v in A:
        assert np.array_equal(A[v]["second"], B[v]["second"]), "encoder and long model rows differ"
        d = {"second": A[v]["second"]}
        key = "probs_steps" if events else "probs_phases"
        pa, pb = A[v][key], B[v][key]
        d[key] = fuse_global(pa, pb, w) if how == "global" else fuse_conf(
            pa, pb, confidence(pa, events), confidence(pb, events))
        out[v] = d
    return out


# ------------------------------------------------------------------------------------ evaluation
class Ctx:
    """Labels, ground truth, priors and ignore windows of one split of fold 0."""

    def __init__(self, st, label_dir=ev.LAB4, ignore_dir=ev.LAB1):
        from eval_untrimmed import ignore_windows

        sp = splits()
        self.set, self.videos, self.train = st, sp[st], sp["train"]
        self.label_dir = label_dir
        self.labs = {v: ev.labels(label_dir, v) for v in self.videos}
        self.ign = ignore_windows(ev.FRAME_DIR, ignore_dir, self.videos)
        self._pri, self._php = {}, None

    def gts(self, names):
        return {v: ev.gt_events(self.labs[v], names) for v in self.videos}

    def priors(self, names):
        k = tuple(names)
        if k not in self._pri:
            self._pri[k] = ev.priors(self.train, list(names), self.label_dir)
        return self._pri[k]

    def phase_priors(self):
        if self._php is None:
            self._php = pd_.fit_priors([ev.labels(self.label_dir, p).phase[::4] for p in self.train], len(F.PHASE_IDS))
        return self._php

    def detect(self, preds, names, variant, fps=4):
        pri = self.priors(names)
        return {v: ev.detect(preds[v], fps, variant, pri, self.ign[v]) for v in self.videos}


def per_video_group_ap(m, n, videos, classes, group_cls):
    out = {}
    for d in DELTAS:
        out[d] = {}
        for v in videos:
            a = [ev.fast_ap(*m[(v, c, d)], n[(v, c)]) for c in group_cls]
            a = [x for x in a if not np.isnan(x)]
            out[d][v] = float(np.mean(a)) if a else float("nan")
    return out


def eval_events(dets, gts, names, videos, n_boot=0, seed=0):
    """eval_focus8.spotting plus ``per_video_group_ap[group][delta][video]`` (for the paired tests)."""
    res = ev.spotting(dets, gts, names, n_boot, seed)
    classes = list(range(1, len(names)))
    m, n = ev.matches(dets, gts, classes)
    pv = {}
    for g, pred in (("short", lambda x: x in F.SHORT), ("long", lambda x: x in F.LONG)):
        gc = [c for c in classes if pred(names[c])]
        if gc:
            pv[g] = {str(d): x for d, x in per_video_group_ap(m, n, videos, classes, gc).items()}
    res["per_video_group_ap"] = pv
    return res


def score(res, group):
    """Selection score: group mAP averaged over +-1/2/5 s."""
    return float(np.nanmean([res["group_map"][group][str(d)] for d in DELTAS]))


def eval_phases(preds, ctx):
    """Phase metrics of ``{video: {second, probs_phases}}`` plus per-video F1@50 / acc and decoded P1."""
    from sklearn.metrics import f1_score

    C, pri = len(F.PHASE_IDS), ctx.phase_priors()
    rt, rp, f1, acc, dec = [], [], {"P0": {}, "P1": {}}, {"P0": {}, "P1": {}}, {}
    for v in ctx.videos:
        z, lab = preds[v], ctx.labs[v]
        rt.append(ev.row_phase(lab, z["second"]))
        rp.append(z["probs_phases"].argmax(1))
        p1s, secs = ev.per_second(z["probs_phases"], z["second"])
        g1 = ev.row_phase(lab, secs)
        d = {"P0": p1s.argmax(1), "P1": pd_.decode_hysteresis(p1s, pri)}
        for k, x in d.items():
            mm = pd_.video_metrics(x, g1, C)
            f1[k][v], acc[k][v] = float(mm["F1@50"]), float(mm["acc"])
        dec[v] = {"secs": secs, "gt": g1, "P0": d["P0"], "P1": d["P1"], "probs": p1s}
    t, p = np.concatenate(rt), np.concatenate(rp)
    agg = {"row_accuracy": float((t == p).mean()), "row_macro_f1": float(f1_score(t, p, average="macro")),
           "F1@50_mean_video": {k: float(np.mean(list(x.values()))) for k, x in f1.items()},
           "acc_1hz_mean_video": {k: float(np.mean(list(x.values()))) for k, x in acc.items()}}
    return agg, {"F1@50": f1, "acc_1hz": acc}, dec


def wilcoxon(x, y):
    """Paired two-sided Wilcoxon signed-rank test on the videos where both values exist."""
    from scipy.stats import wilcoxon as _w

    vs = [v for v in x if v in y and not (np.isnan(x[v]) or np.isnan(y[v]))]
    a, b = np.array([x[v] for v in vs]), np.array([y[v] for v in vs])
    d = a - b
    out = {"n_videos": len(vs), "mean_diff": float(d.mean()) if len(d) else float("nan"),
           "median_diff": float(np.median(d)) if len(d) else float("nan"),
           "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())}
    out["p"] = 1.0 if len(d) == 0 or not np.any(d != 0) else float(_w(a, b).pvalue)
    return out


def match_with_gt(pred_times, pred_scores, gt_times, delta):
    """Greedy one-to-one matching as ``spotting_eval.match_events``, returning ``[(pred index, gt index)]``.

    >>> match_with_gt([10, 11, 30], [0.9, 0.8, 0.5], [10, 50], 2)
    [(0, 0)]
    """
    pt, ps, gt = (np.asarray(x, float) for x in (pred_times, pred_scores, gt_times))
    used, out = np.zeros(len(gt), bool), []
    if len(gt) == 0:
        return out
    for i in np.argsort(-ps, kind="stable"):
        dist = np.abs(gt - pt[i])
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= delta:
            used[j] = True
            out.append((int(i), j))
    return sorted(out)


def best_thresholds(dets, gts, classes, delta=2.0, grid=None):
    """Per-class score threshold with the best F1 at +-delta s (classes without GT keep 0.5).

    >>> d = {"v": [(1, 10.0, 0.9), (1, 50.0, 0.3)]}
    >>> best_thresholds(d, {"v": [(1, 10.5)]}, [1])[1]
    0.35
    """
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2) if grid is None else grid
    out = {}
    for c in classes:
        n_gt = sum(1 for v in gts for k, _ in gts[v] if k == c)
        if n_gt == 0:
            out[c] = 0.5
            continue
        best, best_f = 0.5, -1.0
        for th in grid:
            tp = fp = 0
            for v in gts:
                p = [(t, s) for k, t, s in dets.get(v, []) if k == c and s >= th]
                g = [t for k, t in gts[v] if k == c]
                m, _, _ = se.match_events([t for t, _ in p], [s for _, s in p], g, delta)
                tp += int(m.sum())
                fp += len(p) - int(m.sum())
            f = 2 * tp / max(2 * tp + fp + (n_gt - tp), 1)
            if f > best_f + 1e-12:
                best, best_f = float(th), f
        out[c] = best
    return out


def apply_thresholds(dets, th):
    return {v: [(c, t, s) for c, t, s in d if s >= th.get(c, 0.5)] for v, d in dets.items()}


def dump_json(obj, path):
    """json.dump with NaN/inf as null (the viewer parses the files in a browser)."""
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        return x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(obj), indent=1))


def to_json_dets(dets):
    return {v: [[int(c), float(t), float(s)] for c, t, s in d] for v, d in dets.items()}
