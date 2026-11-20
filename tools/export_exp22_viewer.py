#!/usr/bin/env python3
"""
EXP-22 viewer: static page with the report tables of every fold (results/exp22/fold<i>/report/report.md) and two
figures (group mAP of the models on fold 0 with their bootstrap CIs; L-A3 vs reference per fold). Aggregates only.
Linked from EXP21_focus8/index.html.

    python tools/export_exp22_viewer.py --out ~/workspace/viewer/EXP22_s_long
"""

import argparse
import html
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXP22 = REPO / "results/exp22"


def md_to_html(md):
    """Headings, pipe tables and paragraphs of report.md.

    >>> md_to_html("## a\\n\\n| x | y |\\n|---|---|\\n| 1 | 2 |")
    '<h2>a</h2>\\n<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table>'
    """
    out, rows = [], []

    def flush():
        if rows:
            head, *body = rows
            out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>")
            rows.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            if re.fullmatch(r"\|[-|]+\|", line):
                continue
            rows.append([html.escape(c.strip()) for c in line.strip().strip("|").split(" | ")])
            continue
        flush()
        m = re.match(r"(#+) (.*)", line)
        if m:
            out.append(f"<h{len(m.group(1))}>{html.escape(m.group(2))}</h{len(m.group(1))}>")
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    flush()
    return "\n".join(out)


def fig_fold0(rep, path):
    res = rep["results"]
    names = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, g in zip(axes, ("short", "long")):
        w = 0.8 / len(names)
        for k, n in enumerate(names):
            runs = res[n]["nms"]
            if g not in runs[0]["group_map"]:
                continue
            v = [np.mean([r["group_map"][g][str(d)] for r in runs]) for d in (1, 2, 5)]
            ax.bar(np.arange(3) + k * w, v, w, label=n)
        ax.set_xticks(np.arange(3) + 0.4 - w / 2)
        ax.set_xticklabels(["+-1 s", "+-2 s", "+-5 s"])
        ax.set_title(f"{g} group mAP (nms, mean of seeds, held-out, fold 0)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_cv(reps, path):
    folds = sorted(reps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, g in zip(axes, ("short", "long")):
        a = [np.mean([r["group_map"][g]["2"] for r in reps[f]["results"]["S L-A3 offline"]["nms"]]) for f in folds]
        b = [np.mean([r["group_map"][g]["2"] for r in reps[f]["results"]["S MS-TCN reference"]["nms"]]) for f in folds]
        ax.bar(np.arange(len(folds)) - 0.2, b, 0.4, label="S MS-TCN reference")
        ax.bar(np.arange(len(folds)) + 0.2, a, 0.4, label="S L-A3 offline")
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels([f"fold {f}" for f in folds])
        ax.set_title(f"{g} mAP@+-2 s (nms, mean of 2 seeds)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    reps = {}
    for f in range(5):
        p = EXP22 / f"fold{f}/report/report.json"
        if p.exists():
            reps[f] = json.loads(p.read_text())
    body = ["<h1>EXP-22: L-A3 on the EXP-19 S encoder features</h1>",
            "<p>Fold 0 first (held-out, 10 videos), then the cross-validation check on folds 1-4 (offline L-A3 only). "
            "Aggregates only. See EXPERIMENT.md of the branch exp/focus8-s-long.</p>",
            '<p><a href="../EXP21_focus8/index.html">EXP-21 viewer</a></p>']
    if 0 in reps:
        fig_fold0(reps[0], out / "figures/fold0_group_map.png")
        body.append('<img src="figures/fold0_group_map.png" style="max-width:100%">')
    if len(reps) > 1 and all("S L-A3 offline" in r["results"] for r in reps.values()):
        fig_cv(reps, out / "figures/cv_group_map2.png")
        body.append('<img src="figures/cv_group_map2.png" style="max-width:100%">')
    for f in sorted(reps):
        body.append(md_to_html((EXP22 / f"fold{f}/report/report.md").read_text()))
    css = ("body{font-family:sans-serif;max-width:1300px;margin:1em auto}table{border-collapse:collapse;font-size:12px;margin:8px 0}"
           "td,th{border:1px solid #bbb;padding:2px 6px}th{background:#eee}")
    (out / "index.html").write_text(f"<!doctype html><meta charset=utf-8><title>EXP-22</title><style>{css}</style>" +
                                    "\n".join(body))
    (out / "summary.json").write_text(json.dumps({f: {"prediction_check": r["prediction_check"]} for f, r in reps.items()}, indent=1))
    idx = out.parent / "EXP21_focus8/index.html"
    link = '<p><a href="../EXP22_s_long/index.html">EXP-22: L-A3 on the S features</a></p>'
    if idx.exists() and "EXP22_s_long" not in idx.read_text():
        t = idx.read_text()
        idx.write_text(t.replace("<body>", "<body>" + link, 1) if "<body>" in t else link + t)
    print("viewer written", out, time.strftime("%F %T"))


if __name__ == "__main__":
    main()
