#!/usr/bin/env python3
"""
EXP-21 phase 3, step 5: confusion matrices, offline and online, from the selected predictions of the fold 0
held-out videos.

Events: the 8 targets plus the "Missed" column and the "False alarm" row, computed with
``tools/event_confusion.confusion_counts`` (same-class matching first, then cross-class on the leftovers) at
+-2 s and +-5 s; per-class thresholds = best F1 at +-2 s on the inner val (offline: computed here from
``final_offline.json``; online: the ones stored by ``online_eval_focus8.py``).
Phases: row-normalised counts over the seconds of the 10 videos, 15 classes: offline = the P1-decoded phase of the
selected arm, online = causal hysteresis of the causal long model (and raw argmax P0 in both).

    python tools/confusion_focus8.py --out results/p3/confusion
Writes ``<out>/{offline,online}/events_d<delta>.{csv,json,png}`` and ``phases_<P0|P1>.{csv,json,png}``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F  # noqa: E402

PHASE_LABELS = [f"P{p}" for p in F.PHASE_IDS]


def table(mat, names):
    """Rows: targets + False alarm; columns: targets + Missed (index 0 = background is dropped)."""
    k = len(names)
    keep = list(range(1, k))
    rows = keep + [k]
    return mat[np.ix_(rows, rows)], [names[i] for i in keep]


def phase_matrix(dec, key, n=len(F.PHASE_IDS)):
    """Counts [true, predicted] over the seconds of ``dec = {video: {gt, <key>}}``.

    >>> phase_matrix({"v": {"gt": np.array([0, 0, 1]), "P1": np.array([0, 1, 1])}}, "P1", 2).tolist()
    [[1, 1], [0, 1]]
    """
    m = np.zeros((n, n), int)
    for d in dec.values():
        np.add.at(m, (d["gt"], d[key]), 1)
    return m


def load_phases(path, st="test"):
    z = np.load(path)
    dec = {}
    for k in z.files:
        s, v, f = k.split("/")
        if s == st:
            dec.setdefault(v, {})[f] = z[k]
    return dec


def plot(mat, rows, cols, title, path, normalize=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = mat / np.maximum(mat.sum(1, keepdims=True), 1) if normalize else mat
    fig, ax = plt.subplots(figsize=(0.5 * len(cols) + 4, 0.5 * len(rows) + 3))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for r in range(len(rows)):
        for c in range(len(cols)):
            if mat[r, c]:
                ax.text(c, r, int(mat[r, c]), ha="center", va="center", fontsize=7,
                        color="white" if norm[r, c] > 0.5 else "black")
    ax.set_xticks(range(len(cols)), [x[:28] for x in cols], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)), [x[:28] for x in rows], fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_events(dets, gts, th, out, tag):
    import pandas as pd

    from event_confusion import confusion_counts

    names = F.EVENT_CLASSES
    kept = P.apply_thresholds(dets, th)
    res = {}
    for delta in (2.0, 5.0):
        mat = confusion_counts(kept, gts, delta, len(names), (0,))
        sub, labels = table(mat, names)
        stem = out / f"events_d{delta:g}"
        pd.DataFrame(sub, index=labels + ["False alarm"], columns=labels + ["Missed"]).to_csv(f"{stem}.csv")
        k = len(names)
        s = {"delta": delta, "thresholds": {names[c]: t for c, t in th.items()}, "tp": int(np.trace(mat)),
             "confused": int(mat[:k, :k].sum() - np.trace(mat)), "missed": int(mat[:k, k].sum()),
             "false_alarms": int(mat[k, :k].sum()), "n_gt": int(mat[:k].sum()), "matrix": sub.tolist(),
             "rows": labels + ["False alarm"], "cols": labels + ["Missed"]}
        P.dump_json(s, f"{stem}.json")
        plot(sub, labels + ["False alarm"], labels + ["Missed"],
             f"{tag} events +-{delta:g} s (TP {s['tp']}, confused {s['confused']}, missed {s['missed']}, FA {s['false_alarms']})",
             f"{stem}.png")
        res[delta] = {k_: v for k_, v in s.items() if k_ not in ("matrix", "rows", "cols")}
    return res


def write_phases(dec, key, out, tag):
    mat = phase_matrix(dec, key)
    stem = out / f"phases_{key}"
    np.savetxt(f"{stem}.csv", mat, fmt="%d", delimiter=",", header=",".join(PHASE_LABELS))
    P.dump_json({"matrix": mat.tolist(), "labels": PHASE_LABELS, "accuracy_seconds": float(np.trace(mat) / max(mat.sum(), 1))},
                f"{stem}.json")
    plot(mat, PHASE_LABELS, PHASE_LABELS, f"{tag} phases ({key}), seconds, row-normalised", f"{stem}.png")
    return float(np.trace(mat) / max(mat.sum(), 1))


def main():
    import json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(P.OUT / "confusion"))
    ap.add_argument("--fusion", default=str(P.OUT / "fusion"))
    ap.add_argument("--online", default=str(P.OUT / "online"))
    a = ap.parse_args()
    out = Path(a.out)
    cv, ct = P.Ctx("inner_val"), P.Ctx("held_out")
    gts_v, gts_t = cv.gts(F.EVENT_CLASSES), ct.gts(F.EVENT_CLASSES)
    summary = {}
    fo = Path(a.fusion) / "final_offline.json"
    if fo.exists():
        d = json.loads(fo.read_text())
        dv = {v: [tuple(x) for x in d["inner_val"].get(v, [])] for v in gts_v}
        dt = {v: [tuple(x) for x in d["held_out"].get(v, [])] for v in gts_t}
        th = P.best_thresholds(dv, gts_v, list(range(1, len(F.EVENT_CLASSES))), 2.0)
        (out / "offline").mkdir(parents=True, exist_ok=True)
        summary["offline_events"] = write_events(dt, gts_t, th, out / "offline", "offline")
        ph = Path(a.fusion) / "final_phases.npz"
        if ph.exists():
            dec = load_phases(ph)
            summary["offline_phases_acc_seconds"] = {k: write_phases(dec, k, out / "offline", "offline") for k in ("P0", "P1")}
    on = Path(a.online) / "final_online.json"
    if on.exists():
        d = json.loads(on.read_text())
        th = {F.EVENT_CLASSES.index(n): t for n, t in d["thresholds"].items()}
        dt = {v: [(c, t, s) for c, t, s, _ in d["held_out"].get(v, [])] for v in gts_t}
        (out / "online").mkdir(parents=True, exist_ok=True)
        summary["online_events"] = write_events(dt, gts_t, th, out / "online", "online")
        ph = Path(a.online) / "online_phases.npz"
        if ph.exists():
            dec = load_phases(ph)
            summary["online_phases_acc_seconds"] = {k: write_phases(dec, k, out / "online", "online") for k in ("P0", "P1")}
    P.dump_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=1)[:1500])


if __name__ == "__main__":
    main()
