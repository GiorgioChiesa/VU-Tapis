#!/usr/bin/env python3
"""EXP-25a page: tables (results/exp25a/report.md) and trade-off plots (200 dpi, displayed at the same width as at 100 dpi) -> <out>/index.html.

    python tools/export_exp25a_viewer.py --out ~/workspace/viewer/EXP25a_lookahead
"""
import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402

OUT = P.REPO / "results/exp25a"
FS = (2, 4, 6, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/gchie/workspace/viewer/EXP25a_lookahead")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    R = json.loads((OUT / "report.json").read_text())
    ki = {k: i for i, k in enumerate(R["keys"])}
    lat = R["latency"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, g, ttl in ((axs[0], "mean8", "online hybrid, 8-event mAP@2"), (axs[1], "short", "short group mAP@2")):
        xs = [lat[f"F={F}"]["hybrid|2" if g == "mean8" else "short|2"]["mean"] for F in FS]
        ys = [R["mean_pt"][f"F={F}"][ki[g]][1] for F in FS]
        ci = [[R["mean_ci"][f"F={F}"][i][ki[g]][1] for F in FS] for i in (0, 1)]
        ax.errorbar(xs, ys, yerr=[[y - c for y, c in zip(ys, ci[0])], [c - y for y, c in zip(ys, ci[1])]], marker="o", capsize=3, label="look-ahead F")
        for x, y, F in zip(xs, ys, FS):
            ax.annotate(f"F={F}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=8)
        k0 = "F=0 (EXP-25)"
        x0 = lat[k0]["hybrid|2" if g == "mean8" else "short|2"]["mean"]
        y0 = R["mean_pt"][k0][ki[g]][1]
        ax.errorbar([x0], [y0], yerr=[[y0 - R["mean_ci"][k0][0][ki[g]][1]], [R["mean_ci"][k0][1][ki[g]][1] - y0]], marker="s", color="C3", capsize=3, label="F=0 (EXP-25)")
        for nm, col in (("offline hybrid (EXP-24)", "C2"), ("offline reference (EXP-24)", "C4")):
            if g == "mean8" or nm.startswith("offline hybrid"):
                ax.axhline(R["mean_pt"][nm][ki[g]][1], color=col, ls="--", lw=1, label=nm + " (no latency limit)")
        ax.set_xlabel("mean latency of the true positives at +-2 s (s)")
        ax.set_ylabel("mAP@2 (5-fold mean, 95% CI)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures/tradeoff_map2_vs_latency.png", dpi=200)
    plt.close(fig)
    md = (OUT / "report.md").read_text()
    body = "".join(f"<h2>{html.escape(b.splitlines()[0].lstrip('# '))}</h2><pre>{html.escape(chr(10).join(b.splitlines()[1:]))}</pre>"
                   for b in ["## " + s for s in md.split("\n## ")[1:]])
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>EXP-25a look-ahead</title><style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}"
        "pre{font-size:12px;overflow-x:auto;background:#f6f6f6;padding:8px}img{max-width:100%}</style>"
        "<h1>EXP-25a: look-ahead of F seconds for the short events only</h1>"
        "<p><a href='../EXP25_online/index.html'>EXP-25 online hybrid</a> | <a href='../EXP24_hybrid/index.html'>EXP-24 offline hybrid</a> | "
        "<a href='../timelines_focus8/index.html'>timeline viewer (includes the best F)</a></p>"
        f"<p>Best F (pre-registered inner-val rule): {R['best_F']} s</p>"
        "<img src='figures/tradeoff_map2_vs_latency.png' width='1000'>" + body)
    print("written", out / "index.html")


if __name__ == "__main__":
    main()
