#!/usr/bin/env python3
"""
Event confusion matrix at +/- delta seconds, from the per-second predictions of
`tools/infer_untrimmed.py` (or any predictor that writes the same npz format).

Matching, inside each video:
  1. same-class one-to-one matching within delta (`spotting_eval.match_events`),
     so the diagonal equals the TPs of `spotting_eval.operating_point`;
  2. the still unmatched predictions and ground-truth events are matched again
     ignoring the class (greedy by descending score, nearest unmatched GT within
     delta): these are the confusions (row = true event, column = predicted event);
  3. leftovers: unmatched GT -> column "Missed", unmatched predictions -> row "False alarm".

Only aggregate counts over the requested patients are written (csv, png, json).

Example:
    python tools/event_confusion.py \
        --pred-dir outputs/orsi/LONG/control/totale/untrimmed/checkpoint_best_steps/val \
        --frame-dir /home/gchie/workspace/data/orsi_tensors --delta 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_untrimmed as eu  # noqa: E402  (also puts the repo on sys.path)
from tapis.evaluate import spotting_eval as se  # noqa: E402


def confusion_counts(preds, gts, delta, num_classes, ignore_classes=(0,)):
    """
    Args:
        preds (dict): ``{video: [(class, t, score), ...]}`` (already thresholded).
        gts (dict): ``{video: [(class, t), ...]}``.

    Returns:
        np.ndarray ``[C+1, C+1]``: rows are true classes plus a last "False alarm"
        row, columns are predicted classes plus a last "Missed" column. Ignored
        classes keep empty rows/columns.

    >>> m = confusion_counts({"v": [(1, 10, .9), (2, 20, .8), (1, 50, .7)]},
    ...                      {"v": [(1, 11), (1, 21), (2, 80)]}, 5, 3)
    >>> m[1, 1], m[1, 2], m[2, 3], m[3, 1]
    (1, 1, 1, 1)
    """
    k = num_classes
    mat = np.zeros((k + 1, k + 1), dtype=int)
    for v in set(preds) | set(gts):
        p = [e for e in preds.get(v, []) if e[0] not in ignore_classes]
        g = [e for e in gts.get(v, []) if e[0] not in ignore_classes]
        p_used = np.zeros(len(p), dtype=bool)
        g_used = np.zeros(len(g), dtype=bool)
        # 1. same class
        for c in {e[0] for e in p} | {e[0] for e in g}:
            pi = [i for i, e in enumerate(p) if e[0] == c]
            gi = [j for j, e in enumerate(g) if e[0] == c]
            tp, _, matched = se.match_events([p[i][1] for i in pi], [p[i][2] for i in pi], [g[j][1] for j in gi], delta)
            for i, ok in zip(pi, tp):
                p_used[i] = ok
            for j, ok in zip(gi, matched):
                g_used[j] = ok
            mat[c, c] += int(tp.sum())
        # 2. cross class, on the leftovers
        g_t = np.array([e[1] for e in g], dtype=np.float64)
        for i in sorted(np.flatnonzero(~p_used), key=lambda i: -p[i][2]):
            if not len(g):
                break
            dist = np.abs(g_t - p[i][1])
            dist[g_used] = np.inf
            j = int(np.argmin(dist))
            if dist[j] <= delta:
                mat[g[j][0], p[i][0]] += 1
                p_used[i] = g_used[j] = True
        # 3. leftovers
        for i in np.flatnonzero(~p_used):
            mat[k, p[i][0]] += 1
        for j in np.flatnonzero(~g_used):
            mat[g[j][0], k] += 1
    return mat


def plot(mat, labels, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.42 * n + 4, 0.42 * n + 3))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for r in range(n):
        for c in range(n):
            if mat[r, c]:
                ax.text(c, r, mat[r, c], ha="center", va="center", fontsize=6,
                        color="white" if norm[r, c] > 0.5 else "black")
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--frame-dir", default="/data/orsi_tensors")
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--nms-window", type=float, default=5.0)
    parser.add_argument("--mode", default="nms", choices=["nms", "argmax"])
    parser.add_argument("--threshold", default="best",
                        help="score threshold, or 'best' = the best-F1 threshold at --delta (as in eval_untrimmed)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    meta = json.loads((pred_dir / "meta.json").read_text())
    run = Path(meta["run"])
    cfg = eu.load_cfg(run, meta["ckpt"])
    patients = args.patients or meta["patients"]
    table, split_ids = eu.build_label_table(cfg, patients, args.frame_dir, pred_dir)
    task = "steps" if "steps" in meta["tasks"] else "actions"
    g2l, names = eu.global_to_local(run, task, split_ids[task])
    bg = eu.idle_index(names)

    preds = {p: dict(np.load(pred_dir / f"{p}.npz")) for p in patients}
    durations = {p: float(len(preds[p]["second"])) for p in patients}
    events = table[(table["event_name"].str.strip().str.lower() != "idle") & table["event_id"].isin(g2l)]
    gts = {p: [] for p in patients}
    for row in events.itertuples():
        gts[row.patient_name].append((g2l[row.event_id], float(row.second)))
    dets = {
        p: [(c, float(preds[p]["second"][t]), s)
            for c, t, s in se.decode_events(preds[p][f"probs_{task}"], args.nms_window, 0.0, bg, args.mode)]
        for p in patients
    }
    if args.threshold == "best":
        op = se.best_f1_threshold(dets, gts, args.delta, durations, np.round(np.arange(0.05, 1.0, 0.05), 2), bg)
    else:
        op = se.operating_point(dets, gts, args.delta, durations, float(args.threshold), bg)
    th = op["threshold"]
    kept = {p: [e for e in d if e[2] >= th] for p, d in dets.items()}
    mat = confusion_counts(kept, gts, args.delta, len(names), bg)

    keep = [i for i in range(len(names)) if i not in bg]
    rows = keep + [len(names)]
    labels = [names[i] for i in keep]
    sub = mat[np.ix_(rows, rows)]
    df = pd.DataFrame(sub, index=labels + ["False alarm"], columns=labels + ["Missed"])

    k = len(names)
    tp, conf = int(np.trace(mat)), int(mat[:k, :k].sum() - np.trace(mat))
    summary = {
        "pred_dir": str(pred_dir), "patients": patients, "delta": args.delta, "mode": args.mode,
        "threshold": th, "n_gt": int(mat[:k].sum()), "n_pred": int(mat[:, :k].sum()),
        "tp": tp, "confused": conf, "missed": int(mat[:k, k].sum()), "false_alarms": int(mat[k, :k].sum()),
        "check_vs_operating_point": {"tp": op["tp"], "fp": op["fp"], "n_gt": op["n_gt"]},
        "top_confusions": [
            {"true": names[r], "pred": names[c], "n": int(mat[r, c])}
            for r, c in sorted(((r, c) for r in keep for c in keep if r != c and mat[r, c]), key=lambda x: -mat[x])[:15]
        ],
    }
    assert summary["tp"] == op["tp"] and conf + summary["false_alarms"] == op["fp"], "mismatch with operating_point"

    tag = "" if patients == meta["patients"] else "_" + "-".join(p.removeprefix("RARP") for p in patients)
    stem = pred_dir / f"confusion{tag}_d{args.delta:g}_{args.mode}_th{th:g}"
    df.to_csv(f"{stem}.csv")
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2))
    plot(sub, labels + ["FA / Missed"],
         f"{pred_dir.parent.parent.name}/{pred_dir.parent.name} — ±{args.delta:g}s, {args.mode}, th={th:g} "
         f"(TP {tp}, confused {conf}, missed {summary['missed']}, FA {summary['false_alarms']})", f"{stem}.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_confusions"}, indent=2))
    print(f"Written {stem}.{{csv,json,png}}")


if __name__ == "__main__":
    main()
