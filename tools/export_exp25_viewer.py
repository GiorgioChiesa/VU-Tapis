#!/usr/bin/env python3
"""
EXP-25 page: tables of results/exp25/report.md and 3 aggregate figures (group mAP@2 per fold and 5-fold mean with CI,
AP@2 per class, latency histogram + phases F1@50). Aggregates only, no frames. Linked from EXP24_hybrid/index.html.

    python tools/export_exp25_viewer.py --out ~/workspace/viewer/EXP25_online
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
from export_exp22_viewer import md_to_html  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/exp25"
FOLDS = range(5)
ON, OFFH, REF = "online hybrid", "offline hybrid", "offline reference"
SHORT_ON = "causal short MS-TCN alone (online decoding)"
COL = {ON: "#d95f02", OFFH: "#7570b3", REF: "#1b9e77", SHORT_ON: "#e7298a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "workspace/viewer/EXP25_online"))
    V = Path(ap.parse_args().out)
    (V / "figures").mkdir(parents=True, exist_ok=True)
    rep = json.loads((OUT / "report.json").read_text())
    KEYS = rep["keys"]
    ki = {k: i for i, k in enumerate(KEYS)}
    f25 = {f: json.loads((OUT / f"fold{f}/fold.json").read_text()) for f in FOLDS}
    f24 = {f: json.loads((REPO / f"results/exp24/fold{f}/fold.json").read_text()) for f in FOLDS}
    get = {ON: lambda f: f25[f]["events"][ON], OFFH: lambda f: f24[f]["events"]["hybrid"], REF: lambda f: f24[f]["events"]["reference"],
           SHORT_ON: lambda f: f25[f]["events"][SHORT_ON]}
    x = np.arange(6)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, g in zip(axs, ("mean8", "short", "long")):
        ms = [ON, OFFH, REF] + ([SHORT_ON] if g == "short" else [])
        w = 0.8 / len(ms)
        for i, m in enumerate(ms):
            v = [np.array(get[m](f)["point"])[ki[g]][1] for f in FOLDS] + [np.array(rep["mean_pt"][m])[ki[g]][1]]
            lo = [np.array(get[m](f)["ci95"])[0][ki[g]][1] for f in FOLDS] + [np.array(rep["mean_ci"][m])[0][ki[g]][1]]
            hi = [np.array(get[m](f)["ci95"])[1][ki[g]][1] for f in FOLDS] + [np.array(rep["mean_ci"][m])[1][ki[g]][1]]
            ax.bar(x + (i - (len(ms) - 1) / 2) * w, v, w, color=COL[m], label=m, yerr=[np.array(v) - lo, np.array(hi) - v], capsize=2,
                   ecolor="#555", error_kw={"lw": 0.8})
        ax.set_xticks(x, [f"fold {f}" for f in FOLDS] + ["5-fold mean"], fontsize=8)
        ax.set_title({"mean8": "8-event mean", "short": "short group", "long": "long group"}[g] + " mAP@2 s")
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    axs[1].legend(fontsize=7, loc="lower left"); axs[0].set_ylabel("mAP@2 s (95% bootstrap CI)")
    axs[0].axhline(0.5, color="r", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(V / "figures/group_map2_per_fold.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    xc = np.arange(8)
    for i, m in enumerate([ON, OFFH, REF]):
        v = np.array([rep["mean_pt"][m][ki[c]][1] for c in KEYS[:8]])
        lo = np.array([rep["mean_ci"][m][0][ki[c]][1] for c in KEYS[:8]])
        hi = np.array([rep["mean_ci"][m][1][ki[c]][1] for c in KEYS[:8]])
        ax.bar(xc + (i - 1) * 0.27, v, 0.27, color=COL[m], label=m, yerr=[v - lo, hi - v], capsize=2, ecolor="#555", error_kw={"lw": 0.8})
    ax.set_xticks(xc, [c.replace("_-_right", "").replace("_", " ") for c in KEYS[:8]], rotation=25, ha="right", fontsize=8)
    ax.axvline(3.5, color="k", lw=0.6)
    ax.text(1.5, 1.02, "short", ha="center"); ax.text(5.5, 1.02, "long", ha="center")
    ax.set_ylim(0, 1.08); ax.set_ylabel("AP@2 s, 5-fold mean"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(V / "figures/class_ap2_5fold.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    lat = np.array(rep["latency"]["hybrid|2"]["values"])
    axs[0].hist(lat, bins=np.arange(0, 12.1, 0.25), color=COL[ON])
    axs[0].axvline(lat.mean(), color="k", ls="--", lw=0.8); axs[0].axvline(5, color="r", lw=0.8)
    axs[0].set_title(f"latency of the true positives at +-2 s (mean {lat.mean():.2f} s, std {lat.std():.2f}, n {len(lat)}; red = 5 s)")
    axs[0].set_xlabel("emission time + clip delay - ground truth (s)")
    rows = np.array([[float(c) for c in r[1:]] for r in rep["phases"]])  # 5 folds + mean; columns: onlineP0, onlineH, offH P0, offH P1, ref P0, ref P1, ...
    lab = [("online P0", 0, "#fdae6b"), ("online H", 1, COL[ON]), ("offline hybrid P1", 3, COL[OFFH]), ("offline ref P1", 5, COL[REF])]
    for i, (n, c, col) in enumerate(lab):
        axs[1].bar(x + (i - 1.5) * 0.2, rows[:, c], 0.2, color=col, label=n)
    axs[1].set_xticks(x, [f"fold {f}" for f in FOLDS] + ["mean"]); axs[1].set_title("phases F1@50 (%)"); axs[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(V / "figures/latency_and_phases.png", dpi=130); plt.close(fig)

    md = (OUT / "report.md").read_text()
    (V / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>EXP-25 online hybrid</title><style>body{font-family:sans-serif;max-width:1250px;margin:2em auto}"
        "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #bbb;padding:2px 6px}img{max-width:100%}</style>"
        "<h1>EXP-25: fully online hybrid (causal S MS-TCN short events + causal L-A3 long events and phases)</h1>"
        "<p><a href='../EXP24_hybrid/index.html'>EXP-24 hybrid</a> | <a href='../EXP22_s_long/index.html'>EXP-22</a> | "
        "<a href='../timelines_focus8/index.html'>timeline viewer (EXP-25 online hybrid included)</a></p>"
        "<img src='figures/group_map2_per_fold.png'><img src='figures/class_ap2_5fold.png'><img src='figures/latency_and_phases.png'>" + md_to_html(md))
    print("written", V / "index.html")


if __name__ == "__main__":
    main()
