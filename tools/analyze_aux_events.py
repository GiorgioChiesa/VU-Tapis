#!/usr/bin/env python3
"""
Experiment 2 analysis: which auxiliary class (phase/step/action/instrument/gesture predicted by the
models of `tools/train_aux.py`, trained only on GraSP / SAR-RARP50 / ESAD) signals which Orsi event?

For every pair (auxiliary class a, Orsi event class c), over the 50 Orsi videos:
  AUROC of the posterior of ``a`` separating the seconds within +/- ``--near`` s of an onset of ``c``
  (positives) from the seconds farther than ``--far`` s from any onset of ``c`` (negatives).
0.5 = no information, > 0.5 = the auxiliary class is higher around the event.

Outputs (aggregate only), in --out:
  aux_event_auroc.json   matrix, class names, top auxiliary classes per event and per family
  aux_event_auroc.png    heatmap (events grouped by family, auxiliary classes by source)
  aux_event_profiles.png mean posterior of the best auxiliary class in +/-60 s around the onsets,
                         one panel per event family

Example:
    python tools/analyze_aux_events.py --out outputs/orsi/EXP2/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import rankdata

import train_aux
from summarize_exp2 import FAMILIES, family_of
from train_hybrid import load_orsi

ROOT = Path(__file__).resolve().parents[1]


def auroc(score, pos, neg):
    """Mann-Whitney AUROC of ``score`` for positives vs negatives (boolean masks).

    >>> s = np.array([0.9, 0.8, 0.2, 0.1]); auroc(s, np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool))
    1.0
    """
    x = np.concatenate([score[pos], score[neg]])
    r = rankdata(x)
    n1, n0 = pos.sum(), neg.sum()
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aux-dir", default=str(ROOT / "outputs/orsi/AUX/final"))
    ap.add_argument("--feat-dir", default=str(ROOT / "outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all"))
    ap.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--out", default=str(ROOT / "outputs/orsi/EXP2/analysis"))
    ap.add_argument("--near", type=float, default=3.0)
    ap.add_argument("--far", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ns = SimpleNamespace(feat_dir=args.feat_dir, frame_dir=args.frame_dir, cv_fold=None, target="point", k=3,
                         sigma=2.0, streams="dynamic")
    data, _, info = load_orsi(ns, out)
    names = info["names"]
    events = [c for c, n in enumerate(names) if n.lower() != "idle"]

    # auxiliary posteriors, all tasks, concatenated: [T, A] per video
    aux_cols, aux_src = [], []
    vocab = {}
    for ds in ("grasp", "sar", "esad"):
        rep = json.loads((Path(args.aux_dir) / ds / "aux_report.json").read_text())
        vocab.update(rep["vocab"])
        for t, _, n in rep["groups"]:
            aux_cols += [f"{t}:{v}" for v in rep["vocab"][t]]
            aux_src += [t] * n
    X, ev_time = [], {c: [] for c in events}
    offset = 0
    dist = {c: [] for c in events}
    for p in sorted(data):
        cols = []
        for ds in ("grasp", "sar", "esad"):
            npz = np.load(Path(args.aux_dir) / ds / f"{p}.npz")
            rep_groups = [g[0] for g in json.loads((Path(args.aux_dir) / ds / "aux_report.json").read_text())["groups"]]
            cols += [npz[t].astype(np.float32) for t in rep_groups]
        x = np.concatenate(cols, 1)
        T = len(x)
        tt = np.arange(T)
        for c in events:
            on = np.array([t for cc, t in data[p]["events"] if cc == c])
            d = np.abs(tt[:, None] - on[None]).min(1) if len(on) else np.full(T, np.inf)
            dist[c].append(d)
            ev_time[c] += [offset + int(round(t)) for t in on]
        X.append(x)
        offset += T
    X = np.concatenate(X)
    dist = {c: np.concatenate(v) for c, v in dist.items()}

    A = np.full((len(events), X.shape[1]), np.nan)
    for i, c in enumerate(events):
        pos, neg = dist[c] <= args.near, dist[c] > args.far
        if pos.sum() < 5:
            continue
        rng = np.random.default_rng(0)
        neg_idx = np.flatnonzero(neg)
        if len(neg_idx) > 60000:
            sub = np.zeros_like(neg)
            sub[rng.choice(neg_idx, 60000, replace=False)] = True
            neg = sub
        for j in range(X.shape[1]):
            A[i, j] = auroc(X[:, j], pos, neg)

    ev_names = [names[c] for c in events]
    top = {}
    for i, n in enumerate(ev_names):
        if np.all(np.isnan(A[i])):
            continue
        order = np.argsort(-np.nan_to_num(A[i], nan=0.5))[:3]
        top[n] = [(aux_cols[j], round(float(A[i, j]), 3)) for j in order]
    fam_top = {}
    for f in FAMILIES:
        rows = [i for i, n in enumerate(ev_names) if family_of(n) == f and not np.all(np.isnan(A[i]))]
        if rows:
            m = np.nanmean(A[rows], 0)
            order = np.argsort(-np.nan_to_num(m, nan=0.5))[:5]
            fam_top[f] = [(aux_cols[j], round(float(m[j]), 3)) for j in order]
    src_best = {}
    for s in dict.fromkeys(aux_src):
        cols = [j for j, t in enumerate(aux_src) if t == s]
        src_best[s] = float(np.nanmean(np.nanmax(np.abs(A[:, cols] - 0.5), 1)) + 0.5)
    (out / "aux_event_auroc.json").write_text(json.dumps({
        "near_s": args.near, "far_s": args.far, "events": ev_names, "aux_classes": aux_cols,
        "auroc": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in A],
        "top_per_event": top, "top_per_family": fam_top, "mean_best_auroc_per_source": src_best,
    }, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK2, SURF = "#52514e", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    order_ev = sorted(range(len(ev_names)), key=lambda i: (list(FAMILIES).index(family_of(ev_names[i]))
                                                           if family_of(ev_names[i]) in FAMILIES else 9, ev_names[i]))
    fig, ax = plt.subplots(figsize=(18, 8.5))
    im = ax.imshow(A[order_ev], aspect="auto", cmap="RdBu_r", vmin=0.2, vmax=0.8, interpolation="nearest")
    ax.set_yticks(range(len(order_ev)), [f"[{family_of(ev_names[i])[:5]}] {ev_names[i][:42]}" for i in order_ev], fontsize=6.5)
    short = [c.split(":", 1)[1][:22] for c in aux_cols]
    ax.set_xticks(range(len(aux_cols)), short, rotation=90, fontsize=5.5)
    b = 0
    for s in dict.fromkeys(aux_src):
        n = aux_src.count(s)
        ax.axvline(b - 0.5, color=INK2, lw=0.8)
        ax.text(b + n / 2 - 0.5, -1.2, s, ha="center", fontsize=7.5, color=INK2)
        b += n
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="AUROC (0.5 = nessuna informazione)")
    ax.set_title("Quale classe ausiliaria segnala quale evento Orsi (AUROC, ±3 s dall'onset contro >30 s)", pad=18)
    fig.tight_layout()
    fig.savefig(out / "aux_event_auroc.png", dpi=150)
    plt.close(fig)

    # profiles: best auxiliary class per family, mean posterior around the onsets of that family
    fig, axs = plt.subplots(1, len(fam_top), figsize=(4 * len(fam_top), 3.2), sharey=False)
    w = np.arange(-60, 61)
    for ax, (f, best) in zip(np.atleast_1d(axs), fam_top.items()):
        for k, (cname, auc) in enumerate(best[:3]):
            j = aux_cols.index(cname)
            onsets = [t for i, c in enumerate(events) if family_of(names[c]) == f for t in ev_time[c]]
            prof = np.array([X[np.clip(t + w, 0, len(X) - 1), j] for t in onsets]).mean(0)
            base = X[:, j].mean()
            ax.plot(w, prof, lw=2, color=["#2a78d6", "#eb6834", "#1baf7a"][k], label=f"{cname.split(':', 1)[1][:24]} ({auc:.2f})")
            ax.axhline(base, color=["#2a78d6", "#eb6834", "#1baf7a"][k], ls=":", lw=1)
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_title(f.replace("_", " "))
        ax.set_xlabel("secondi dall'onset")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    np.atleast_1d(axs)[0].set_ylabel("posterior media (punteggiato = media globale)")
    fig.tight_layout()
    fig.savefig(out / "aux_event_profiles.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"top_per_family": fam_top, "mean_best_auroc_per_source": src_best}, indent=1))


if __name__ == "__main__":
    main()
