#!/usr/bin/env python3
"""
EXP-25a summary over the 5 folds and the 4 look-aheads: raw tables (md + json), pooled Wilcoxon, prediction check, latency, best F.
Reads results/exp25a/fold<i>/eval_F<F>/{fold.json, boot.npz}, results/exp25/fold<i>/{fold.json, boot.npz} (F = 0) and
results/exp24/fold<i>/{fold.json, boot.npz} (offline hybrid / reference); writes results/exp25a/report.{md,json}.

    python tools/exp25a_summary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402

OUT, O25, O24 = P.REPO / "results/exp25a", P.REPO / "results/exp25", P.REPO / "results/exp24"
FOLDS, FS, DELTAS = range(5), (2, 4, 6, 10), list(P.DELTAS)


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def cell(pt, ci, j):
    return f"{pt[j]:.3f} [{ci[0][j]:.2f}, {ci[1][j]:.2f}]"


def main():
    fj = {F: {f: json.loads((OUT / f"fold{f}/eval_F{F}/fold.json").read_text()) for f in FOLDS} for F in FS}
    f25 = {f: json.loads((O25 / f"fold{f}/fold.json").read_text()) for f in FOLDS}
    f24 = {f: json.loads((O24 / f"fold{f}/fold.json").read_text()) for f in FOLDS}
    KEYS = f25[0]["keys"]
    ki = {k: i for i, k in enumerate(KEYS)}
    # models: name -> (fold json dict, event key, boot dict per fold)
    boots = {F: {f: dict(np.load(OUT / f"fold{f}/eval_F{F}/boot.npz")) for f in FOLDS} for F in FS}
    b25 = {f: dict(np.load(O25 / f"fold{f}/boot.npz")) for f in FOLDS}
    b24 = {f: dict(np.load(O24 / f"fold{f}/boot.npz")) for f in FOLDS}
    M = {}
    for F in FS:
        M[f"F={F}"] = {"ev": {f: fj[F][f]["events"]["online hybrid"] for f in FOLDS}, "bt": {f: boots[F][f]["online_hybrid"] for f in FOLDS}}
        M[f"F={F} short alone"] = {"ev": {f: fj[F][f]["events"]["short part alone (online decoding)"] for f in FOLDS},
                                   "bt": {f: boots[F][f]["short_part_alone_(online_decoding)"] for f in FOLDS}}
    M["F=0 (EXP-25)"] = {"ev": {f: f25[f]["events"]["online hybrid"] for f in FOLDS}, "bt": {f: b25[f]["online_hybrid"] for f in FOLDS}}
    M["F=0 short alone (EXP-25)"] = {"ev": {f: f25[f]["events"]["causal short MS-TCN alone (online decoding)"] for f in FOLDS},
                                     "bt": {f: b25[f]["causal_short_MS-TCN_alone_(online_decoding)"] for f in FOLDS}}
    M["offline hybrid (EXP-24)"] = {"ev": {f: f24[f]["events"]["hybrid"] for f in FOLDS}, "bt": {f: b24[f]["hybrid"] for f in FOLDS}}
    M["offline reference (EXP-24)"] = {"ev": {f: f24[f]["events"]["reference"] for f in FOLDS}, "bt": {f: b24[f]["reference"] for f in FOLDS}}
    names = list(M)
    pt = {m: {f: np.array(M[m]["ev"][f]["point"]) for f in FOLDS} for m in names}
    mean_pt = {m: np.mean([pt[m][f] for f in FOLDS], 0) for m in names}
    mean_bt = {m: np.mean([M[m]["bt"][f] for f in FOLDS], 0) for m in names}
    mean_ci = {m: np.nanpercentile(mean_bt[m], [2.5, 97.5], axis=0) for m in names}
    hyb = ["F=0 (EXP-25)"] + [f"F={F}" for F in FS] + ["offline hybrid (EXP-24)", "offline reference (EXP-24)"]
    md = ["# EXP-25a: short-event look-ahead F (online hybrid with a short MS-TCN that sees F s of the future), 5 folds x 10 held-out videos\n",
          "AP in [0, 1]; CI = per-video bootstrap (1000, same resamples for all models), 5-fold mean CI = bootstrap of the mean of the 5 folds. "
          "Latency = emission time + F (short part) + clip delay 2 s - ground truth; the decoder delay (2 s) is inside the emission time.\n"]
    sel = []
    for F in FS:
        for f in FOLDS:
            s = fj[F][f]["selection"]
            sel.append([F, f, s["short_chosen"].replace("|", " / "), f"{s['short_inner_val_score']:.3f}", s["long_chosen_exp25"].replace("|", " / ")])
    md += ["## Choices made on the inner val (short part cand / decoder, its inner-val score; long part = EXP-25 choice)",
           table(["F", "fold", "short part", "inner short score", "long part"], sel), ""]
    for g, title in (("short", "short group mAP"), ("mean8", "8-event mean mAP (online hybrid)"), ("long", "long group mAP")):
        rows = []
        for f in FOLDS:
            for m in hyb:
                p, c = pt[m][f], np.nanpercentile(M[m]["bt"][f], [2.5, 97.5], axis=0)
                rows.append([f, m] + [cell(p[ki[g]], c[:, ki[g]], j) for j in range(3)])
        for m in hyb:
            rows.append(["5-fold mean", m] + [cell(mean_pt[m][ki[g]], mean_ci[m][:, ki[g]], j) for j in range(3)])
        md += [f"## {title}, AP@1 / @2 / @5 s per fold and 5-fold mean", table(["fold", "model", "@1", "@2", "@5"], rows), ""]
    rows = []
    for g in ("short", "mean8"):
        for m in hyb:
            rows.append([g, m] + [f"{pt[m][f][ki[g]][1]:.3f}" for f in FOLDS] + [f"{mean_pt[m][ki[g]][1]:.3f}"])
    md += ["## Summary, mAP@2 per fold and 5-fold mean", table(["group", "model", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean"], rows), ""]
    # short alone
    rows = [[m] + [cell(mean_pt[m][ki["short"]], mean_ci[m][:, ki["short"]], j) for j in range(3)]
            for m in ["F=0 short alone (EXP-25)"] + [f"F={F} short alone" for F in FS]]
    md += ["## Short part alone (online decoding), short-group mAP, 5-fold mean", table(["model", "@1", "@2", "@5"], rows), ""]
    # per class (5-fold mean @2)
    rows = [[c] + [f"{mean_pt[m][ki[c]][1]:.3f}" for m in hyb] for c in KEYS[:8]]
    md += ["## AP@2 per class, 5-fold mean", table(["class"] + hyb, rows), ""]
    # differences with CI
    rows = []
    for F in FS:
        for ref in ("F=0 (EXP-25)", "offline hybrid (EXP-24)"):
            d = mean_bt[f"F={F}"] - mean_bt[ref]
            ci = np.nanpercentile(d, [2.5, 97.5], axis=0)
            for g in ("mean8", "short"):
                rows.append([f"F={F} - {ref}", g] + [cell((mean_pt[f"F={F}"] - mean_pt[ref])[ki[g]], ci[:, ki[g]], j) for j in range(3)])
    md += ["## Differences, 5-fold mean (bootstrap CI)", table(["comparison", "group", "@1", "@2", "@5"], rows), ""]
    # latency
    lat = {F: {k: np.concatenate([fj[F][f]["latency_tp"][k] for f in FOLDS]) for k in fj[F][0]["latency_tp"]} for F in FS}
    lat0 = {k: np.concatenate([f25[f]["latency_tp"][k] for f in FOLDS]) for k in f25[0]["latency_tp"]}
    rows = []
    for nm, L in [("F=0 (EXP-25)", lat0)] + [(f"F={F}", lat[F]) for F in FS]:
        for k in ("hybrid|2", "short|2", "short|1", "short|5", "long|2"):
            x = L[k]
            rows.append([nm, k.replace("|", " +-"), f"{x.mean():.2f}", f"{x.std():.2f}", len(x)])
    md += ["## Latency of the true positives (s) pooled over the 5 folds (hybrid = all 8 classes)", table(["model", "part tolerance", "mean", "std", "n TP"], rows), ""]
    # Wilcoxon pooled
    def pooled(m, g, d):
        out = {}
        for f in FOLDS:
            out.update(M[m]["ev"][f]["per_video"][g][str(d)])
        return out
    wil, rows = {}, []
    for F in FS:
        for ref in ("F=0 (EXP-25)", "offline hybrid (EXP-24)"):
            for g in ("mean8", "short"):
                w = P.wilcoxon(pooled(f"F={F}", g, 2), pooled(ref, g, 2))
                wil[f"F={F} - {ref}|{g}|2"] = w
                rows.append([f"F={F} - {ref}", g, "@2", w["n_videos"], f"{w['mean_diff']:+.3f}", f"{w['n_better']}/{w['n_worse']}", f"{w['p']:.4g}"])
    md += ["## Pooled paired Wilcoxon over the 50 videos (per-video mAP@2; 8 tests per metric, Bonferroni threshold 0.006)",
           table(["comparison", "group", "tol", "n", "mean diff", "better/worse", "p"], rows), ""]
    # best F (rule fixed in the pre-registration: smallest F within 0.02 of the maximum of the 5-fold mean inner-val short score)
    inner = {F: float(np.mean([fj[F][f]["selection"]["short_inner_val_score"] for f in FOLDS])) for F in FS}
    best = min(F for F in FS if inner[F] >= max(inner.values()) - 0.02)
    md += ["## Best F (pre-registered rule: smallest F within 0.02 of the maximum 5-fold mean inner-val short score)",
           table(["F", "mean inner-val short score"], [[F, f"{inner[F]:.3f}"] for F in FS]), f"\nbest F = {best} s\n"]
    # prediction
    s6, m6 = mean_pt["F=6"][ki["short"]][1], mean_pt["F=6"][ki["mean8"]][1]
    pred = {"short_group_at2_F6": float(s6), "holds_short_ge_0.65": bool(s6 >= 0.65), "hybrid_mean8_at2_F6": float(m6), "holds_mean8_ge_0.50": bool(m6 >= 0.50),
            "latency_mean_at2_F6": float(lat[6]["hybrid|2"].mean())}
    md += ["## Prediction check (pre-registered in EXPERIMENT.md, EXP-25a)",
           table(["quantity", "value"], [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in pred.items()]), ""]
    text = "\n".join(md)
    (OUT / "report.md").write_text(text)
    P.dump_json({"keys": KEYS, "models": names, "mean_pt": {m: mean_pt[m] for m in names}, "mean_ci": {m: mean_ci[m] for m in names},
                 "fold_pt": {m: {str(f): pt[m][f] for f in FOLDS} for m in names}, "wilcoxon": wil, "prediction": pred, "best_F": best,
                 "inner_val_short_score": inner,
                 "latency": {**{f"F={F}": {k: {"mean": float(x.mean()), "std": float(x.std()), "n": len(x)} for k, x in lat[F].items()} for F in FS},
                             "F=0 (EXP-25)": {k: {"mean": float(x.mean()), "std": float(x.std()), "n": len(x)} for k, x in lat0.items()}}},
                OUT / "report.json")
    print(text)


if __name__ == "__main__":
    main()
