#!/usr/bin/env python3
"""
Esp6a diagnostics on posteriors in the shared format (aggregates only; nothing per video is printed).

  * ground truth of every video of ``--gt-dir``: per-phase duration table (presence, segments,
    median / min / p5 / p25 / p75 / max segment duration), Idle position (start / interior / end),
    phase order (modal order, most frequent sequence, skips, repeats, inverted pairs, transition
    types, Kendall tau of the first-occurrence order vs the modal order);
  * oracle-phase arm (``--oracle name=dir`` ...): accuracy, per-phase recall, share of the errors
    per true phase, accuracy in the first / last 10 % of each video vs the middle; ``--fold`` keeps
    one outer fold (to compare the fold-0 CPU reruns with the GPU run).

Usage:
    python tools/exp6_diagnostics.py --gt-dir outputs/orsi/EXP6/F_base/seed0/phase_posteriors \\
        --oracle gpu=outputs/orsi/EXP6/oracle_2.3/seed0/phase_posteriors \\
                 start=outputs/orsi/EXP6/oracle_crop/start/seed0/phase_posteriors \\
                 center=outputs/orsi/EXP6/oracle_crop/center/seed0/phase_posteriors \\
        --fold 0 --out outputs/orsi/EXP6/diagnostics.json
"""

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_decode import load_posteriors, modal_order, runs  # noqa: E402

IDLE = 0


def phase_sequence(gt):
    """Non-Idle phase sequence of a video, with consecutive repeats (split by Idle) collapsed."""
    seq = [c for c, _, _ in runs(gt[gt >= 0]) if c != IDLE]
    return [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]


def duration_table(gts, names):
    rows = []
    for c, n in enumerate(names):
        d = [e - s for g in gts for cc, s, e in runs(g[g >= 0]) if cc == c]
        pres = sum(any(cc == c for cc, _, _ in runs(g[g >= 0])) for g in gts)
        tot = [int((g == c).sum()) for g in gts if (g == c).any()]
        q = np.percentile(d, [0, 5, 25, 50, 75, 100]).tolist() if d else [None] * 6
        rows.append({"phase": n, "presence": pres, "n_videos": len(gts), "segments": len(d),
                     "seg_per_present_video": len(d) / pres if pres else None,
                     **dict(zip(["min", "p5", "p25", "median", "p75", "max"], q)),
                     "total_per_video_median": float(np.median(tot)) if tot else None})
    return rows


def idle_position(gts):
    pos = Counter()
    secs = Counter()
    for g in gts:
        seg = runs(g[g >= 0])
        for i, (c, s, e) in enumerate(seg):
            if c != IDLE:
                continue
            where = "start" if i == 0 else "end" if i == len(seg) - 1 else "interior"
            pos[where] += 1
            secs[where] += e - s
    tot = sum(secs.values())
    return {w: {"segments": pos[w], "seconds_share": secs[w] / tot if tot else None}
            for w in ("start", "interior", "end")}


def order_stats(gts, names):
    C = len(names)
    seqs = [phase_sequence(g) for g in gts]
    order = [c for c in modal_order(seqs, C) if c != IDLE]
    rank = {c: i for i, c in enumerate(order)}
    nonidle = [c for c in range(C) if c != IDLE]
    seq_counts = Counter(tuple(s) for s in seqs)
    skips, repeats, inv_pairs, ntrans = Counter(), Counter(), Counter(), Counter()
    n_skips, n_rep, n_inv, taus, exact, strict = [], [], [], [], 0, 0
    for s in seqs:
        present = set(s)
        miss = [c for c in nonidle if c not in present]
        skips.update(miss)
        n_skips.append(len(miss))
        cnt = Counter(s)
        rep = [c for c, k in cnt.items() if k > 1]
        repeats.update(rep)
        n_rep.append(len(rep))
        first = list(dict.fromkeys(s))  # first occurrences, in time order
        inv = [(a, b) for a, b in itertools.combinations(first, 2) if rank[a] > rank[b]]  # a first, against the order
        inv_pairs.update(inv)
        n_inv.append(len(inv))
        if len(first) > 2:
            taus.append(kendalltau(range(len(first)), [rank[c] for c in first]).statistic)
        exact += (not inv and not rep)
        strict += (not inv and not rep and not miss)
        for a, b in zip(s[:-1], s[1:]):
            d = rank[b] - rank[a]
            ntrans["forward_next" if d == 1 else "forward_skip" if d > 1 else "backward"] += 1
    nm = lambda c: names[c]
    return {
        "modal_order": [nm(c) for c in order],
        "n_videos": len(seqs),
        "distinct_sequences": len(seq_counts),
        "most_frequent_sequence_count": seq_counts.most_common(1)[0][1],
        "videos_exact_modal_order": strict,
        "videos_in_modal_order_allowing_skips": exact,
        "skips_per_video": dict(sorted(Counter(n_skips).items())),
        "most_skipped": [(nm(c), k) for c, k in skips.most_common(6)],
        "videos_with_repeats": int(sum(x > 0 for x in n_rep)),
        "most_repeated": [(nm(c), k) for c, k in repeats.most_common(6)],
        "videos_with_inversions": int(sum(x > 0 for x in n_inv)),
        "inversions_per_video": dict(sorted(Counter(n_inv).items())),
        "most_inverted_pairs": [(f"{nm(a)} before {nm(b)}", k) for (a, b), k in inv_pairs.most_common(8)],
        "transitions": dict(ntrans),
        "kendall_tau_mean": float(np.mean(taus)), "kendall_tau_min": float(np.min(taus)),
        "distinct_transitions": len({(a, b) for s in seqs for a, b in zip(s[:-1], s[1:])}),
    }


def oracle_stats(post_dir, names, fold=None, edge=0.1):
    videos, _ = load_posteriors(post_dir)
    if fold is not None:
        videos = {v: d for v, d in videos.items() if d["fold"] == fold}
    C = len(names)
    conf = np.zeros((C, C), dtype=np.int64)
    pos = {"edge": [0, 0], "middle": [0, 0]}
    per_fold = {}
    for d in videos.values():
        g, p = d["gt"], d["probs"].argmax(1)
        keep = g >= 0
        np.add.at(conf, (g[keep], p[keep]), 1)
        T = len(g)
        rel = np.arange(T) / max(1, T - 1)
        ok = (g == p)[keep]
        e = ((rel < edge) | (rel > 1 - edge))[keep]
        pos["edge"][0] += int(ok[e].sum()); pos["edge"][1] += int(e.sum())
        pos["middle"][0] += int(ok[~e].sum()); pos["middle"][1] += int((~e).sum())
        f = per_fold.setdefault(d["fold"], [0, 0])
        f[0] += int(ok.sum()); f[1] += int(keep.sum())
    tp = np.diag(conf)
    sup = conf.sum(1)
    err = sup - tp
    return {"n_videos": len(videos), "acc": float(tp.sum() / conf.sum()),
            "acc_per_fold": {int(k): v[0] / v[1] for k, v in sorted(per_fold.items())},
            "acc_edge_10pct": pos["edge"][0] / max(1, pos["edge"][1]),
            "acc_middle": pos["middle"][0] / max(1, pos["middle"][1]),
            "recall": {names[c]: (float(tp[c] / sup[c]) if sup[c] else None) for c in range(C)},
            "error_share": {names[c]: float(err[c] / err.sum()) for c in range(C) if err[c]},
            "support_share": {names[c]: float(sup[c] / sup.sum()) for c in range(C)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--oracle", nargs="*", default=[], help="name=phase_posteriors dir")
    ap.add_argument("--fold", type=int, help="outer fold for the oracle comparison (default: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    videos, names = load_posteriors(args.gt_dir)
    gts = [d["gt"] for d in videos.values()]
    res = {"durations": duration_table(gts, names), "idle_position": idle_position(gts),
           "order": order_stats(gts, names), "oracle": {}}
    for spec in args.oracle:
        k, d = spec.split("=", 1)
        res["oracle"][k] = {"all": oracle_stats(d, names)}
        if args.fold is not None:
            res["oracle"][k][f"fold{args.fold}"] = oracle_stats(d, names, args.fold)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["order"], indent=0)[:3000])


if __name__ == "__main__":
    main()
