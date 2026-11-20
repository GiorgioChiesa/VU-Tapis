#!/usr/bin/env python3
"""
EXP-25 summary over the 5 folds: raw tables (md + json), pooled Wilcoxon, prediction check, latency, phases.
Reads results/exp25/fold<i>/{fold.json, boot.npz} and results/exp24/fold<i>/{fold.json, boot.npz}; writes
results/exp25/report.{md,json}. The page and the figures are made by tools/export_exp25_viewer.py from report.json.

    python tools/exp25_summary.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402

OUT, O24 = P.REPO / "results/exp25", P.REPO / "results/exp24"
FOLDS = range(5)
DELTAS = list(P.DELTAS)
ON, OFFH, REF = "online hybrid", "offline hybrid", "offline reference"
SHORT_ON, SHORT_NMS = "causal short MS-TCN alone (online decoding)", "causal short MS-TCN alone (nms decoding, offline decoder)"
MODELS = [ON, OFFH, REF, SHORT_ON, SHORT_NMS]


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def cell(pt, ci, j):
    return f"{pt[j]:.3f} [{ci[0][j]:.2f}, {ci[1][j]:.2f}]"


def main():
    f25 = {f: json.loads((OUT / f"fold{f}/fold.json").read_text()) for f in FOLDS}
    f24 = {f: json.loads((O24 / f"fold{f}/fold.json").read_text()) for f in FOLDS}
    KEYS = f25[0]["keys"]
    ki = {k: i for i, k in enumerate(KEYS)}
    b25 = {f: dict(np.load(OUT / f"fold{f}/boot.npz")) for f in FOLDS}
    b24 = {f: dict(np.load(O24 / f"fold{f}/boot.npz")) for f in FOLDS}
    ev_pt, ev_bt, ev_pv = {}, {}, {}
    src = {ON: ("25", ON), OFFH: ("24", "hybrid"), REF: ("24", "reference"), SHORT_ON: ("25", SHORT_ON), SHORT_NMS: ("25", SHORT_NMS)}
    for m, (e, k) in src.items():
        fj, bb = (f25, b25) if e == "25" else (f24, b24)
        ev_pt[m] = {f: np.array(fj[f]["events"][k]["point"]) for f in FOLDS}
        ev_bt[m] = {f: bb[f][k.replace(" ", "_")] for f in FOLDS}
        ev_pv[m] = {f: fj[f]["events"][k]["per_video"] for f in FOLDS}
    mean_pt = {m: np.mean([ev_pt[m][f] for f in FOLDS], 0) for m in MODELS}
    mean_bt = {m: np.mean([ev_bt[m][f] for f in FOLDS], 0) for m in MODELS}
    mean_ci = {m: np.nanpercentile(mean_bt[m], [2.5, 97.5], axis=0) for m in MODELS}
    pt_ci = lambda m, f: (ev_pt[m][f], np.nanpercentile(ev_bt[m][f], [2.5, 97.5], axis=0))
    md = ["# EXP-25: fully online hybrid vs offline hybrid vs offline reference, 5 folds x 10 held-out videos (50 videos)\n",
          "AP in [0, 1]; 95% CI = per-video bootstrap within fold (1000, same resamples for every model), 5-fold mean CI = bootstrap of the mean of the 5 folds. "
          "Online = decoder delay 2 s + clip delay 2 s. Offline hybrid / reference = EXP-24 (reference = mean of its 2 seeds, nms).\n"]
    sel = [[f, f25[f]["selection"]["short_chosen"].replace("|", " / "), f"{f25[f]['selection']['short_inner_val'][f25[f]['selection']['short_chosen']]:.3f}",
            f25[f]["selection"]["long_chosen"].replace("|", " / "), f"{f25[f]['selection']['long_inner_val'][f25[f]['selection']['long_chosen']]:.3f}",
            f25[f]["selection"]["phase_cand"], f25[f]["selection"]["short_nms_chosen"]] for f in FOLDS]
    md += ["## Choices made on the inner val", table(["fold", "short part (cand / decoder)", "inner short score", "long part (cand / decoder)",
                                                       "inner long score", "phases from", "short nms cand"], sel), ""]
    for g in ("mean8", "short", "long"):
        title = {"short": "short group mAP", "long": "long group mAP", "mean8": "8-event mean (mean of the 8 class APs)"}[g]
        rows = []
        for f in FOLDS:
            for m in (ON, OFFH, REF, SHORT_ON) if g != "long" else (ON, OFFH, REF):
                if m == SHORT_ON and g == "mean8":
                    continue
                p, c = pt_ci(m, f)
                rows.append([f, m] + [cell(p[ki[g]], c[:, ki[g]], j) for j in range(3)])
        for m in (ON, OFFH, REF, SHORT_ON, SHORT_NMS) if g == "short" else ((ON, OFFH, REF) if g == "long" else (ON, OFFH, REF)):
            rows.append(["5-fold mean", m] + [cell(mean_pt[m][ki[g]], mean_ci[m][:, ki[g]], j) for j in range(3)])
        for a_, b_ in ((ON, OFFH), (ON, REF)) + (((SHORT_ON, REF),) if g == "short" else ()):
            d = mean_bt[a_] - mean_bt[b_]
            rows.append(["5-fold mean", f"{a_} - {b_}"] + [cell((mean_pt[a_] - mean_pt[b_])[ki[g]], np.nanpercentile(d, [2.5, 97.5], axis=0)[:, ki[g]], j) for j in range(3)])
        md += [f"## {title} (AP@1 / @2 / @5 s) per fold and 5-fold mean", table(["fold", "model", "@1", "@2", "@5"], rows), ""]
    # compact per-fold @2 table
    rows = []
    for m in (ON, OFFH, REF):
        rows.append([m] + [f"{ev_pt[m][f][ki['mean8']][1]:.3f}" for f in FOLDS] + [f"{mean_pt[m][ki['mean8']][1]:.3f}"])
    rows.append(["short group: causal short alone (online)"] + [f"{ev_pt[SHORT_ON][f][ki['short']][1]:.3f}" for f in FOLDS] + [f"{mean_pt[SHORT_ON][ki['short']][1]:.3f}"])
    rows.append(["short group: offline reference"] + [f"{ev_pt[REF][f][ki['short']][1]:.3f}" for f in FOLDS] + [f"{mean_pt[REF][ki['short']][1]:.3f}"])
    md += ["## Summary, mAP@2: 8-event mean and short group, per fold and 5-fold mean", table(["model", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean"], rows), ""]
    # per class
    rows = []
    for c in KEYS[:8]:
        for m in (ON, OFFH, REF):
            rows.append([c, m, " / ".join(f"{ev_pt[m][f][ki[c]][1]:.3f}" for f in FOLDS)] + [cell(mean_pt[m][ki[c]], mean_ci[m][:, ki[c]], j) for j in range(3)])
        if c in KEYS[:4]:
            rows.append([c, SHORT_ON, " / ".join(f"{ev_pt[SHORT_ON][f][ki[c]][1]:.3f}" for f in FOLDS)] + [cell(mean_pt[SHORT_ON][ki[c]], mean_ci[SHORT_ON][:, ki[c]], j) for j in range(3)])
    md += ["## AP per class: per-fold AP@2 (folds 0-4) and 5-fold mean AP@1 / @2 / @5 with CI", table(["class", "model", "AP@2 folds 0 / 1 / 2 / 3 / 4", "@1", "@2", "@5"], rows), ""]

    # latency
    lat = {k: np.concatenate([f25[f]["latency_tp"][k] for f in FOLDS]) for k in f25[0]["latency_tp"]}
    rows = []
    for k, x in lat.items():
        nm, d = k.split("|")
        rows.append([nm, f"+-{d}", f"{x.mean():.2f}", f"{x.std():.2f}", len(x)] + [f"{np.mean(f25[f]['latency_tp'][k]):.2f}" for f in FOLDS])
    md += ["## Latency of the true positives (s) = emission time + clip delay (2 s) - ground-truth time, pooled over the 5 folds",
           table(["part", "tolerance", "mean", "std", "n TP", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4"], rows), ""]

    # Wilcoxon pooled
    def pooled(m, g, d):
        out = {}
        for f in FOLDS:
            out.update(ev_pv[m][f][g][str(d)])
        return out
    wil = {}
    rows = []
    for a_, b_ in ((ON, OFFH), (ON, REF)):
        for g in ("mean8", "short", "long"):
            for d in DELTAS:
                w = P.wilcoxon(pooled(a_, g, d), pooled(b_, g, d))
                wil[f"{a_} - {b_}|{g}|{d}"] = w
                rows.append([f"{a_} - {b_}", g, f"@{d}", w["n_videos"], f"{w['mean_diff']:+.3f}", f"{w['n_better']}/{w['n_worse']}", f"{w['p']:.4f}"])
    w = P.wilcoxon(pooled(SHORT_ON, "short", 2), pooled(REF, "short", 2))
    wil[f"{SHORT_ON} - {REF}|short|2"] = w
    rows.append([f"{SHORT_ON} - {REF}", "short", "@2", w["n_videos"], f"{w['mean_diff']:+.3f}", f"{w['n_better']}/{w['n_worse']}", f"{w['p']:.4f}"])
    md += ["## Pooled paired Wilcoxon over the 50 videos (per-video group mAP; mean diff, better/worse, p)", table(["comparison", "group", "tol", "n", "mean diff", "better/worse", "p"], rows), ""]

    # phases
    rows, pvals = [], {"online P0": {}, "online H": {}, "offline hybrid P0": {}, "offline hybrid P1": {}, "offline reference P0": {}, "offline reference P1": {}}
    for f in FOLDS:
        ph = f25[f]["phases"]
        a = ph["online"]["agg"]
        rows.append([f, f"{a['F1@50_mean_video']['P0']:.1f}", f"{a['F1@50_mean_video']['H']:.1f}", f"{ph['offline_hybrid'][0]['F1@50_P0']:.1f}",
                     f"{ph['offline_hybrid'][0]['F1@50_P1']:.1f}", f"{ph['offline_reference'][0]['F1@50_P0']:.1f}", f"{ph['offline_reference'][0]['F1@50_P1']:.1f}",
                     f"{a['acc_1hz_mean_video']['P0']:.3f}", f"{a['acc_1hz_mean_video']['H']:.3f}"])
        pvals["online P0"].update(ph["online"]["per_video"]["P0"]); pvals["online H"].update(ph["online"]["per_video"]["H"])
        for nm, k in (("offline hybrid", 1), ("offline reference", 1)):
            for q in ("P0", "P1"):
                pvals[f"{nm} {q}"].update(ph["offline_hybrid" if "hybrid" in nm else "offline_reference"][1][f"F1@50_{q}"])
    rows.append(["mean"] + [f"{np.mean([float(r[i]) for r in rows]):.1f}" for i in range(1, 7)] + [f"{np.mean([float(r[i]) for r in rows]):.3f}" for i in (7, 8)])
    md += ["## Phases (causal L-A3 online vs offline), F1@50 (mean over videos) and 1 Hz accuracy",
           table(["fold", "online P0 (raw)", "online H (causal hysteresis)", "offline hybrid P0", "offline hybrid P1", "offline ref P0", "offline ref P1", "online acc P0", "online acc H"], rows), ""]
    prow = []
    for a_, b_ in (("online H", "offline hybrid P1"), ("online H", "offline reference P1"), ("online P0", "offline hybrid P0")):
        w = P.wilcoxon(pvals[a_], pvals[b_])
        wil[f"phases {a_} - {b_}"] = w
        prow.append([f"{a_} - {b_}", w["n_videos"], f"{w['mean_diff']:+.1f}", f"{w['n_better']}/{w['n_worse']}", f"{w['p']:.4f}"])
    md += [table(["phases comparison (F1@50, 50 videos)", "n", "mean diff", "better/worse", "p"], prow), ""]

    # prediction check
    m8 = mean_pt[ON][ki["mean8"]][1]
    off8 = mean_pt[OFFH][ki["mean8"]][1]
    sh = mean_pt[SHORT_ON][ki["short"]][1]
    rf = mean_pt[REF][ki["short"]][1]
    lm = float(lat["hybrid|2"].mean())
    pred = {"online_hybrid_mean8_at2": m8, "offline_hybrid_mean8_at2": off8, "threshold_0.85x_offline": 0.85 * off8, "ratio": m8 / off8,
            "holds_mean8": bool(m8 >= 0.85 * off8), "latency_mean_at2": lm, "holds_latency": bool(lm <= 5.0),
            "causal_short_at2": sh, "reference_short_at2": rf, "secondary_diff": sh - rf, "holds_secondary": bool(sh >= rf - 0.10)}
    md += ["## Prediction check (pre-registered in EXPERIMENT.md, EXP-25)",
           table(["quantity", "value"], [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in pred.items()]), ""]
    text = "\n".join(md)
    (OUT / "report.md").write_text(text)
    P.dump_json({"keys": KEYS, "mean_pt": {m: mean_pt[m] for m in MODELS}, "mean_ci": {m: mean_ci[m] for m in MODELS},
                 "fold_pt": {m: {str(f): ev_pt[m][f] for f in FOLDS} for m in MODELS}, "wilcoxon": wil, "prediction": pred,
                 "latency": {k: {"mean": float(x.mean()), "std": float(x.std()), "n": len(x), "values": x.tolist()} for k, x in lat.items()},
                 "phases": rows, "selection": {str(f): f25[f]["selection"] for f in FOLDS}}, OUT / "report.json")
    print(text)


if __name__ == "__main__":
    main()
