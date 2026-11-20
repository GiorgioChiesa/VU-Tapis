#!/usr/bin/env python3
"""
EXP-24 summary over the 5 folds: tables (md + json), pooled Wilcoxon, prediction check, figures and the page
~/workspace/viewer/EXP24_hybrid/index.html. Reads results/exp24/fold<i>/{fold.json, boot.npz}.

    python tools/exp24_summary.py
"""

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402

OUT = P.REPO / "results/exp24"
FOLDS = range(5)
MODELS = ["hybrid", "reference", "L-A3 (EXP-22 rule)"]
COL = {"hybrid": "#d95f02", "reference": "#1b9e77", "L-A3 (EXP-22 rule)": "#7570b3"}
DELTAS = list(P.DELTAS)
GROUP_KEYS = ["short", "long", "mean8"]


def f3(x):
    return "-" if x is None or np.isnan(x) else f"{x:.3f}"


def cell(pt, ci, j):
    return f"{pt[j]:.3f} [{ci[0][j]:.2f}, {ci[1][j]:.2f}]"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", default=str(Path.home() / "workspace/viewer/EXP24_hybrid"))
    a = ap.parse_args()
    fj = {f: json.loads((OUT / f"fold{f}/fold.json").read_text()) for f in FOLDS}
    for r in fj.values():  # phase_summary returned (summary, per_video) -> stored as a 2-list
        r["phases"] = {m: {**s, "per_video": pv} for m, (s, pv) in r["phases"].items()}
    KEYS = fj[0]["keys"]
    ki = {k: i for i, k in enumerate(KEYS)}
    boots = {f: dict(np.load(OUT / f"fold{f}/boot.npz")) for f in FOLDS}
    bk = lambda m: m.replace(" ", "_")
    pt = {m: np.mean([np.array(fj[f]["events"][m]["point"]) for f in FOLDS], 0) for m in MODELS}      # [K,3] 5-fold mean
    bt = {m: np.mean([boots[f][bk(m)] for f in FOLDS], 0) for m in MODELS}                              # [B,K,3]
    ci = {m: np.nanpercentile(bt[m], [2.5, 97.5], axis=0) for m in MODELS}
    dpt = pt["hybrid"] - pt["reference"]
    dci = np.nanpercentile(bt["hybrid"] - bt["reference"], [2.5, 97.5], axis=0)
    md = ["# EXP-24: hybrid (reference S MS-TCN short events + L-A3 long events and phases) vs reference vs L-A3, 5 folds x 10 held-out videos\n",
          "AP in [0, 1]; 95% CI = per-video bootstrap within fold (1000, same resamples for all models), 5-fold mean = mean of the 5 fold values.\n"]
    sel = [[f, fj[f]["selection"]["chosen"].replace("|", " / "), f"{fj[f]['selection']['inner_val_long_score'][fj[f]['selection']['chosen']]:.3f}",
            fj[f]["selection"]["phase_cand"], f"{fj[f]['short_diff_max']:.1e}"] for f in FOLDS]
    md += ["## Choices made on the inner val", table(["fold", "L-A3 long events (seed or ens / decoder)", "inner-val long score",
                                                       "phases from", "max |hybrid - reference| short AP"], sel), ""]

    # ---- groups per fold and mean
    for g in GROUP_KEYS:
        rows = []
        for f in FOLDS:
            r = fj[f]["events"]
            for m in MODELS:
                rows.append([f, m] + [cell(np.array(r[m]["point"])[ki[g]], np.array(r[m]["ci95"])[:, ki[g]], j) for j in range(3)])
            rows.append([f, "hybrid - reference"] + [cell(np.array(r["hybrid_minus_reference"]["point"])[ki[g]],
                                                          np.array(r["hybrid_minus_reference"]["ci95"])[:, ki[g]], j) for j in range(3)])
        for m in MODELS:
            rows.append(["mean", m] + [cell(pt[m][ki[g]], ci[m][:, ki[g]], j) for j in range(3)])
        rows.append(["mean", "hybrid - reference"] + [cell(dpt[ki[g]], dci[:, ki[g]], j) for j in range(3)])
        title = {"short": "short group mAP", "long": "long group mAP", "mean8": "8-event mean (mean of the 8 class APs)"}[g]
        md += [f"## {title} (AP@1 / @2 / @5 s) per fold and 5-fold mean", table(["fold", "model", "@1", "@2", "@5"], rows), ""]

    # ---- per class
    rows = []
    for c in KEYS[:8]:
        for m in MODELS:
            rows.append([c, m] + [" / ".join(f3(fj[f]["events"][m]["point"][ki[c]][1]) for f in FOLDS)] +
                        [cell(pt[m][ki[c]], ci[m][:, ki[c]], j) for j in range(3)])
        rows.append([c, "hybrid - reference", ""] + [cell(dpt[ki[c]], dci[:, ki[c]], j) for j in range(3)])
    md += ["## AP per class: per-fold AP@2 (folds 0-4) and 5-fold mean AP@1 / @2 / @5 with CI",
           table(["class", "model", "AP@2 folds 0 / 1 / 2 / 3 / 4", "@1", "@2", "@5"], rows), ""]
    n_ge = {d: sum(dpt[ki[c]][j] >= -1e-12 for c in KEYS[:8]) for j, d in enumerate(DELTAS)}
    md += ["Classes (of 8) with hybrid AP >= reference AP (5-fold mean): " + ", ".join(f"@{d} s: {n}" for d, n in n_ge.items()), ""]

    # ---- phases
    pk = ["row_accuracy", "row_macro_f1", "F1@50_P0", "F1@50_P1", "acc1hz_P0", "acc1hz_P1"]
    rows = []
    for f in FOLDS:
        for m in MODELS:
            rows.append([f, m + (" (phases = L-A3 selected)" if m == "hybrid" else "")] + [f3(fj[f]["phases"][m][k]) if "F1" not in k else f"{fj[f]['phases'][m][k]:.1f}" for k in pk])
    for m in MODELS:
        rows.append(["mean", m] + [(f"{np.mean([fj[f]['phases'][m][k] for f in FOLDS]):.3f}" if "F1" not in k else f"{np.mean([fj[f]['phases'][m][k] for f in FOLDS]):.1f}") for k in pk])
    md += ["## Phases, held-out (row = per-row argmax; P0 = per-second argmax, P1 = hysteresis; F1@50 in percent, mean over videos)",
           table(["fold", "model", "row acc", "row macro F1", "F1@50 P0", "F1@50 P1", "acc 1 Hz P0", "acc 1 Hz P1"], rows), ""]

    # ---- Wilcoxon pooled over the 50 videos
    wil = {}
    for g in GROUP_KEYS:
        for d in DELTAS:
            x, y = {}, {}
            for f in FOLDS:
                x.update(fj[f]["events"]["hybrid"]["per_video"][g][str(d)])
                y.update(fj[f]["events"]["reference"]["per_video"][g][str(d)])
            wil[f"events|{g}|@{d}"] = P.wilcoxon({k: (np.nan if v is None else v) for k, v in x.items()}, {k: (np.nan if v is None else v) for k, v in y.items()})
    for k in ("F1@50_P0", "F1@50_P1", "acc_1hz_P0", "acc_1hz_P1"):
        x, y = {}, {}
        for f in FOLDS:
            x.update(fj[f]["phases"]["hybrid"]["per_video"][k])
            y.update(fj[f]["phases"]["reference"]["per_video"][k])
        wil[f"phases|{k}"] = P.wilcoxon(x, y)
    md += ["## Paired Wilcoxon over the pooled 50 held-out videos, hybrid minus reference (per-video AP of the group, seed mean per video)",
           table(["test", "n videos", "mean diff", "median diff", "hybrid better", "hybrid worse", "p (two-sided)"],
                 [[k, t["n_videos"], f"{t['mean_diff']:+.3f}", f"{t['median_diff']:+.3f}", t["n_better"], t["n_worse"], f"{t['p']:.4f}"] for k, t in wil.items()]),
           "(per-video AP skips classes without a true event in the video; for short, the hybrid and reference values are identical, so n better = n worse = 0)", ""]

    # ---- prediction check
    m8 = dpt[ki["mean8"]][1]
    pred = {"mean8_map2_hybrid": float(pt["hybrid"][ki["mean8"]][1]), "mean8_map2_reference": float(pt["reference"][ki["mean8"]][1]),
            "difference": float(m8), "threshold": 0.02, "holds_mean8": bool(m8 >= 0.02),
            "short_diff_map2": float(dpt[ki["short"]][1]), "holds_short": bool(dpt[ki["short"]][1] >= -0.01),
            "per_fold_mean8_diff_map2": {str(f): float(np.array(fj[f]["events"]["hybrid_minus_reference"]["point"])[ki["mean8"]][1]) for f in FOLDS}}
    md += ["## Prediction check (pre-registered in EXPERIMENT.md, EXP-24)",
           f"8-event mean mAP@2, 5-fold mean: hybrid {pred['mean8_map2_hybrid']:.4f}, reference {pred['mean8_map2_reference']:.4f}, "
           f"difference {m8:+.4f} (threshold +0.02, 95% CI of the difference [{dci[0][ki['mean8']][1]:+.3f}, {dci[1][ki['mean8']][1]:+.3f}]); "
           f"short group difference {pred['short_diff_map2']:+.4f} (limit -0.01). Per fold mean8 difference @2: "
           + ", ".join(f"{k}: {v:+.3f}" for k, v in pred["per_fold_mean8_diff_map2"].items()), ""]

    # ---- online
    if "online" in fj[0]:
        o = fj[0]["online"]
        rows = []
        for name, r in o["results"].items():
            for g in GROUP_KEYS:
                rows.append([name, g] + [cell(np.array(r["point"])[ki[g]], np.array(r["ci95"])[:, ki[g]], j) for j in range(3)])
        r22 = json.loads((P.REPO / "results/exp22/fold0/report/report.json").read_text())["online"]["test_events"]["group_map"]
        for g in ("short", "long"):
            rows.append(["EXP-22 causal L-A3 (short and long both causal)", g] + [f3(r22[g][str(d)]) for d in DELTAS])
        md += [f"## Online, fold 0 only (held-out 10 videos): causal L-A3 long head ({o['chosen']}, selected on inner val) + reference short events "
               "as offline detections with the declared 2 s delay (not a true online number)",
               table(["model", "group", "@1", "@2", "@5"], rows), "",
               "Latency of the long-event true positives (s, emit time + clip delay 2 - ground truth): " + "; ".join(
                   f"+-{d}: mean {l['mean']:.2f} std {l['std']:.2f} n {l['n_tp']}" for d, l in o["latency_long_tp"].items()),
               "Short events: declared delay 2 s, no latency measured.", ""]

    # ---- figures
    V = Path(a.viewer)
    (V / "figures").mkdir(parents=True, exist_ok=True)
    x = np.arange(6)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, g in zip(axs, GROUP_KEYS):
        for i, m in enumerate(MODELS):
            vals = [fj[f]["events"][m]["point"][ki[g]][1] for f in FOLDS] + [pt[m][ki[g]][1]]
            lo = [fj[f]["events"][m]["ci95"][0][ki[g]][1] for f in FOLDS] + [ci[m][0][ki[g]][1]]
            hi = [fj[f]["events"][m]["ci95"][1][ki[g]][1] for f in FOLDS] + [ci[m][1][ki[g]][1]]
            ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=COL[m], label=m, yerr=[np.array(vals) - lo, np.array(hi) - vals], capsize=2, ecolor="#555", error_kw={"lw": 0.8})
        ax.set_xticks(x, [f"fold {f}" for f in FOLDS] + ["5-fold mean"])
        ax.set_title({"short": "short group", "long": "long group", "mean8": "8-event mean"}[g] + " mAP@2 s")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("mAP@2 s (error bars: 95% bootstrap CI)")
    axs[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(V / "figures/group_map2_per_fold.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    xc = np.arange(8)
    for i, m in enumerate(MODELS):
        v = np.array([pt[m][ki[c]][1] for c in KEYS[:8]])
        lo = np.array([ci[m][0][ki[c]][1] for c in KEYS[:8]])
        hi = np.array([ci[m][1][ki[c]][1] for c in KEYS[:8]])
        ax.bar(xc + (i - 1) * 0.27, v, 0.27, color=COL[m], label=m, yerr=[v - lo, hi - v], capsize=2, ecolor="#555", error_kw={"lw": 0.8})
    ax.set_xticks(xc, [c.replace("_-_right", "").replace("_", " ") for c in KEYS[:8]], rotation=25, ha="right", fontsize=8)
    ax.axvline(3.5, color="k", lw=0.6)
    ax.text(1.5, 1.02, "short", ha="center"); ax.text(5.5, 1.02, "long", ha="center")
    ax.set_ylim(0, 1.08); ax.set_ylabel("AP@2 s, 5-fold mean"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(V / "figures/class_ap2_5fold.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for j, d in enumerate(DELTAS):
        d_f = [np.array(fj[f]["events"]["hybrid_minus_reference"]["point"])[ki["mean8"]][j] for f in FOLDS] + [dpt[ki["mean8"]][j]]
        lo = [np.array(fj[f]["events"]["hybrid_minus_reference"]["ci95"])[0][ki["mean8"]][j] for f in FOLDS] + [dci[0][ki["mean8"]][j]]
        hi = [np.array(fj[f]["events"]["hybrid_minus_reference"]["ci95"])[1][ki["mean8"]][j] for f in FOLDS] + [dci[1][ki["mean8"]][j]]
        axs[0].bar(x + (j - 1) * 0.27, d_f, 0.27, label=f"@{d} s", yerr=[np.array(d_f) - lo, np.array(hi) - d_f], capsize=2, error_kw={"lw": 0.8})
    axs[0].axhline(0.02, color="r", ls="--", lw=0.8); axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_xticks(x, [f"fold {f}" for f in FOLDS] + ["5-fold mean"]); axs[0].set_title("8-event mean AP, hybrid minus reference (red: +0.02)")
    axs[0].legend(fontsize=8)
    for i, m in enumerate(MODELS):
        axs[1].bar(x + (i - 1) * 0.27, [fj[f]["phases"][m]["F1@50_P1"] for f in FOLDS] + [np.mean([fj[f]["phases"][m]["F1@50_P1"] for f in FOLDS])],
                   0.27, color=COL[m], label=m)
    axs[1].set_xticks(x, [f"fold {f}" for f in FOLDS] + ["5-fold mean"]); axs[1].set_title("phases F1@50 with hysteresis (P1), %"); axs[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(V / "figures/diff_and_phases.png", dpi=130); plt.close(fig)

    text = "\n".join(md)
    (OUT / "report.md").write_text(text)
    P.dump_json({"events_5fold_mean": {m: pt[m].tolist() for m in MODELS}, "ci95_5fold": {m: ci[m].tolist() for m in MODELS},
                 "hybrid_minus_reference": {"point": dpt.tolist(), "ci95": dci.tolist()}, "wilcoxon": wil, "prediction": pred,
                 "folds": {str(f): {"events": {m: fj[f]["events"][m]["point"] for m in MODELS}, "phases": fj[f]["phases"],
                                    "selection": fj[f]["selection"]} for f in FOLDS}, "keys": KEYS}, OUT / "report.json")
    body = "".join(f"<h2>{html.escape(b.splitlines()[0].lstrip('# '))}</h2><pre>{html.escape(chr(10).join(b.splitlines()[1:]))}</pre>" for b in
                   ("\n".join(md).split("\n## ")[0:1] + ["## " + s for s in "\n".join(md).split("\n## ")[1:]]))
    (V / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>EXP-24 hybrid</title><style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}"
        "pre{font-size:12px;overflow-x:auto;background:#f6f6f6;padding:8px}img{max-width:100%}</style>"
        "<h1>EXP-24: hybrid of the reference S MS-TCN (short events) and L-A3 on the S features (long events, phases)</h1>"
        "<p><a href='../EXP22_s_long/index.html'>EXP-22 viewer</a> | <a href='../timelines_focus8/index.html'>timeline viewer (includes the hybrid)</a> | <a href='../EXP25_online/index.html'>EXP-25 online hybrid</a></p>"
        "<img src='figures/group_map2_per_fold.png'><img src='figures/class_ap2_5fold.png'><img src='figures/diff_and_phases.png'>" + body)
    print(text)


if __name__ == "__main__":
    main()
