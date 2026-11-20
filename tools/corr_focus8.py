#!/usr/bin/env python3
"""
EXP-21 phase 3, step 6: cross-correlation figures between the 8 target events and phases, instruments, actions.

Method of ``analisi/leonardo_action_sota/orsi/orsi_corr.py`` (not edited, it is an untracked top-level script):
per video, every signal is z-scored over the seconds; the Pearson r between the event impulse train (1 at the
second of each event) and the signal shifted by a lag in [-30, +30] s is ``mean z[s + lag] * sqrt(p / (1 - p))``
over the event occurrences, p = events / seconds (pooled over videos); the p-value of the peak |r| comes from
``--nshuf`` circular shifts of the events inside every video. Positive lag = the signal comes after the event.

Two sets, each with the same figures:
  * ``true``: true events and phases (revised labels v2026-11-19) and the TAPIS teacher (instruments 7, actions 14)
    on every video with a teacher file (descriptive; no model);
  * ``pred``: the predicted heads on the 10 held-out videos: events = detections of ``final_offline.json`` above the
    per-class thresholds, phases / instruments / actions = the per-second probabilities of arm M (phases: the selected
    fusion arm) and ``true_test10`` = the ``true`` set restricted to the same 10 videos.
Figures per set: events x phases, events x instruments, events x actions (peak r, the number is the lag in s where
p < 0.01) and the lagged cross-correlation (+-30 s) of each of the 8 events (top 5 signals by |peak r|).
No phases x instruments, phases x actions or instruments x actions.

    python tools/corr_focus8.py --out results/p3/corr
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F, ev  # noqa: E402

ACT_DIR = Path("/home/gchie/workspace/nas_private/data/Orsi/action_labels")
MAX_LAG = 30


def zscore(x):
    """Column-wise z-score over seconds, NaN -> 0.

    >>> zscore(np.array([[1.0], [3.0], [np.nan]])).round(2).tolist()
    [[-1.0], [1.0], [0.0]]
    """
    m, s = np.nanmean(x, 0), np.nanstd(x, 0)
    return np.nan_to_num((x - m) / np.where(s > 0, s, 1.0), nan=0.0)


def xcorr_events(events, signals, n_classes, max_lag=MAX_LAG, nshuf=100, seed=0):
    """
    Args:
        events: ``{video: {class index: [second, ...]}}``; signals: ``{video: [S, K]}``.
    Returns:
        r ``[n_classes, K, 2 * max_lag + 1]``, p ``[n_classes, K]`` (peak |r| against circular shifts), n events.

    >>> sig = {"v": np.zeros((100, 1))}; sig["v"][57, 0] = 1.0
    >>> r, p, n = xcorr_events({"v": {0: [50]}}, sig, 1, max_lag=10, nshuf=5)
    >>> r.shape, int(np.abs(r[0, 0]).argmax()) - 10, n.tolist()
    ((1, 1, 21), 7, [1.0])
    """
    rng = np.random.default_rng(seed)
    lags = np.arange(-max_lag, max_lag + 1)
    Z = {v: zscore(s) for v, s in signals.items()}
    n_tot = sum(len(z) for z in Z.values())
    K = next(iter(Z.values())).shape[1]

    def peri(shift=None):
        acc, cnt = np.zeros((n_classes, len(lags), K)), np.zeros(n_classes)
        for v, per_class in events.items():
            nv = len(Z[v])
            for c, secs in per_class.items():
                ss = np.asarray(secs, int)
                if len(ss) == 0:
                    continue
                if shift is not None:
                    ss = (ss + shift[v]) % nv
                acc[c] += Z[v][np.clip(ss[:, None] + lags[None, :], 0, nv - 1)].sum(0)
                cnt[c] += len(ss)
        return acc, cnt

    acc, cnt = peri()
    p = np.maximum(cnt, 1) / n_tot
    scale = np.sqrt(p / (1 - p))[:, None, None]
    r = (acc / np.maximum(cnt, 1)[:, None, None] * scale).transpose(0, 2, 1)
    peak = np.abs(r).max(2)
    null = np.zeros((nshuf, n_classes, K))
    for b in range(nshuf):
        a2, _ = peri({v: int(rng.integers(len(Z[v]))) for v in Z})
        null[b] = (np.abs(a2 / np.maximum(cnt, 1)[:, None, None] * scale)).max(1)
    pval = (1 + (null >= peak[None]).sum(0)) / (nshuf + 1)
    return r, pval, cnt


def heat(r, pval, lags, rows, cols, title, path, alpha=0.01):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pk = np.abs(r).argmax(2)
    m = np.take_along_axis(r, pk[..., None], 2)[..., 0]
    vm = max(0.02, float(np.abs(m).max()))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols) + 4.5), 0.42 * len(rows) + 2.4))
    ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if pval[i, j] < alpha:
                ax.text(j, i, f"{lags[pk[i, j]]:+d}", ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if abs(m[i, j]) > 0.6 * vm else "black")
    ax.set_xticks(range(len(cols)), [c[:26] for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), [x[:34] for x in rows], fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return m.tolist(), [[int(lags[k]) for k in row] for row in pk]


def lagged(r, lags, rows, cols, title, path, top=5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
    for i, ax in enumerate(axes.ravel()[: len(rows)]):
        idx = np.argsort(-np.abs(r[i]).max(1))[:top]
        for k in idx:
            ax.plot(lags, r[i, k], label=cols[k][:22], lw=1.2)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(rows[i][:34], fontsize=8)
        ax.legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("lag (s), + = signal after the event")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def event_seconds(lab):
    out = {}
    for t, k in lab.events:
        if k > 0 and not lab.ignore[min(lab.n - 1, int(round(t * lab.fps)))]:
            out.setdefault(k - 1, []).append(int(round(t)))
    return out


def true_set(videos, label_dir=ev.LAB4):
    ev_, ph, ins, act = {}, {}, {}, {}
    for v in videos:
        lab = ev.labels(label_dir, v)
        n = lab.n // 4 + 1
        a, i = F.teacher_seconds(ACT_DIR / f"{v}_actions.json", n)
        ev_[v] = {c: [s for s in secs if s < n] for c, secs in event_seconds(lab).items()}
        ph[v] = np.eye(len(F.PHASE_IDS))[lab.phase[::4][:n]]
        a, i = a[:n], i[:n]
        act[v], ins[v] = a, i
    return ev_, {"phases": ph, "instruments": ins, "actions": act}


def pred_set(videos, fusion_dir):
    d = json.loads((Path(fusion_dir) / "final_offline.json").read_text())
    ctx = P.Ctx("inner_val")
    gts_v = ctx.gts(F.EVENT_CLASSES)
    th = P.best_thresholds({v: [tuple(x) for x in d["inner_val"].get(v, [])] for v in gts_v}, gts_v,
                           list(range(1, len(F.EVENT_CLASSES))), 2.0)
    z = np.load(Path(fusion_dir) / "final_phases.npz")
    ev_, ph, ins, act = {}, {}, {}, {}
    for v in videos:
        e = {}
        for c, t, s in d["held_out"].get(v, []):
            if s >= th.get(c, 0.5):
                e.setdefault(c - 1, []).append(int(round(t)))
        enc = dict(np.load(P.ENC_DIR / f"{v}.npz"))
        a, secs = ev.per_second(enc["probs_tapis_actions"], enc["second"])
        i, _ = ev.per_second(enc["probs_tapis_instruments"], enc["second"])
        n = len(secs)
        ev_[v] = {c: [s for s in x if s < n] for c, x in e.items()}
        ph[v], act[v], ins[v] = z[f"test/{v}/probs"][:n], a[:n], i[:n]
    return ev_, {"phases": ph, "instruments": ins, "actions": act}


def make_figures(tag, events, sigs, out, nshuf):
    names = F.TARGETS
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    cols = {"phases": [f"P{p}" for p in F.PHASE_IDS], "instruments": F.INSTRUMENTS, "actions": F.ACTIONS}
    res, all_r, all_cols = {}, [], []
    n_ev = None
    for kind in ("phases", "instruments", "actions"):
        vids = sorted(events)
        r, p, n = xcorr_events(events, {v: sigs[kind][v] for v in vids}, len(names), nshuf=nshuf)
        n_ev = n
        m, lg = heat(r, p, lags, [f"{x} (n={int(n[i])})" for i, x in enumerate(names)], cols[kind],
                     f"{tag}: events x {kind}, peak cross-correlation (lag in s where p < 0.01)",
                     out / f"xcorr_events_{kind}_{tag}.png")
        res[kind] = {"r_peak": m, "lag": lg, "p": p.tolist(), "cols": cols[kind]}
        all_r.append(r)
        all_cols += [f"{kind[:3]}:{c}" for c in cols[kind]]
    lagged(np.concatenate(all_r, 1), lags, names, all_cols,
           f"{tag}: lagged cross-correlation (+-{MAX_LAG} s) of each target event, top 5 signals", out / f"xcorr_lagged_{tag}.png")
    res["n_events"] = {names[i]: int(n_ev[i]) for i in range(len(names))}
    res["n_videos"] = len(events)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(P.OUT / "corr"))
    ap.add_argument("--fusion", default=str(P.OUT / "fusion"))
    ap.add_argument("--nshuf", type=int, default=100)
    ap.add_argument("--videos", type=int, default=0, help="smoke test: first N videos")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = P.splits()
    allv = sorted(p.name[:-13] for p in ACT_DIR.glob("RARP*_actions.json"))
    held = sp["held_out"]
    lim = (lambda x: x[: a.videos]) if a.videos else (lambda x: x)
    summary = {}
    e, s = true_set(lim(allv))
    summary["true"] = make_figures("true", e, s, out, a.nshuf)
    e10, s10 = true_set(lim(held))
    summary["true_test10"] = make_figures("true_test10", e10, s10, out, a.nshuf)
    if (Path(a.fusion) / "final_offline.json").exists():
        e, s = pred_set(lim(held), a.fusion)
        summary["pred"] = make_figures("pred", e, s, out, a.nshuf)
    P.dump_json(summary, out / "corr.json")
    (out / ".done").write_text("ok")
    print({k: v["n_events"] for k, v in summary.items()})


if __name__ == "__main__":
    main()
