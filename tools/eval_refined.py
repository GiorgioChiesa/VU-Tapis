#!/usr/bin/env python3
"""
Experiment 5: spotting metrics of the refined detections (`tools/train_refiner.py`) against F_base.

For every arm (R0 = F_base unchanged, R1-1fps, R1) and seed of ``--fold``, from
``EXP5/<arm>/seed<s>/fold<i>/val/detections.json``:
  * spotting mAP@1/2/3/5, tight average-mAP (1-5 s) and best F1@5 s (same threshold grid as
    `train_temporal.quick_eval`), with the integer ground truth (CSV seconds, as every experiment:
    primary) and with the fractional one (`tools/event_times.py`: secondary);
  * localisation error of the best-scored same-class detection within +/- 5 s of each onset:
    share <= 1 s, <= 2 s, > 2 s, mean signed error (s);
  * references: ``R0_direct`` (F_base npz decoded here: must equal R0), ``R0_rowsec`` (each time
    replaced by the second of its feature row: the 1 fps tables repeat a second shared by two
    events, which shifts every later row by one) and ``oracle`` (every detection refined by the arm
    moved to the nearest same-class onset within +/- 6 s, snapped to 0.25 s: perfect refiner).
Aggregate output only: ``EXP5/pilot_fold<i>.json``, with the gate of the pilot (mAP@1 +0.03 over R0
for every seed and R1 > R1-1fps).

``--cv ARM``: every (seed, fold) of ``--folds``, R0 vs ARM (and, with ``--cc``, R0_cc vs ARM_cc: the
detections of the caps + order-chain decoding, `train_refiner.py --stage refine-cc`); per-run metrics,
paired two-sided Wilcoxon over the runs (delta, p, wins) and the decision rule fixed in advance (tight
avg-mAP +0.03, mAP@1 up with p < 0.05, mAP@5 down by at most 0.01), in ``EXP5/cv_<ARM>.json``.

Example:
    python tools/eval_refined.py --fold 0/5 --seeds 0 1
    python tools/eval_refined.py --cv R1-1fps --folds 0 1 2 3 4 --seeds 0 1 --cc
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from train_refiner import EXP, ROOT, fbase_detections, fold_meta, load_event_times, snap
from tapis.evaluate import spotting_eval as se

DELTAS = (1, 2, 3, 4, 5)
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def loc_errors(dets, gts, tol=5.0):
    """
    Signed error (s) of the best-scored same-class detection within ``tol`` s of each onset.

    >>> loc_errors([(1, 4.5, 0.9), (1, 7, 0.95), (2, 3, 0.5)], [(1, 5.0), (2, 9.0)])
    [2.0]
    """
    out = []
    for c, t in gts:
        cand = [(s, tt) for cc, tt, s in dets if cc == c and abs(tt - t) <= tol]
        if cand:
            out.append(float(max(cand)[1] - t))
    return out


def metrics(dets, gts, lengths, n_classes, bg):
    res = se.spotting_map(dets, gts, DELTAS, n_classes, ignore_classes=bg)
    op = se.best_f1_threshold(dets, gts, 5, lengths, THRESHOLDS, bg)
    e = np.array([x for p in gts for x in loc_errors(dets.get(p, []), gts[p])])
    return {
        **{f"mAP@{d}": float(res["map"][d]) for d in (1, 2, 3, 5)},
        "avg_mAP_tight": se.average_map(res["map"], DELTAS),
        "F1@5": float(op["f1"]),
        "err_n": int(len(e)),
        "err<=1": float(np.mean(np.abs(e) <= 1)),
        "err<=2": float(np.mean(np.abs(e) <= 2)),
        "err>2": float(np.mean(np.abs(e) > 2)),
        "err_mean_signed": float(e.mean()),
        "err_mean_abs": float(np.abs(e).mean()),
    }


def oracle(dets, ref_dets, gts_frac, per_second=4, half=6.0):
    """Detections that the arm refined (time changed or not, score >= min) moved to the nearest true onset."""
    out = {}
    for p, dd in dets.items():
        g = gts_frac.get(p, [])
        new = []
        for (c, t, s), (_, t0, _) in zip(dd, ref_dets[p]):
            cand = [tt for cc, tt in g if cc == c and abs(tt - t0) <= half]
            if s >= 0.1 and cand:
                t = snap(min(cand, key=lambda x: abs(x - t0)), per_second)
            new.append((c, t, s))
        out[p] = new
    return out


WILCOXON_KEYS = ("mAP@1", "mAP@2", "avg_mAP_tight", "mAP@5", "F1@5", "err<=1")


def load_dets(arm, seed, fi):
    f = EXP / arm / f"seed{seed}" / f"fold{fi}" / "val" / "detections.json"
    return {p: [tuple(d) for d in dd] for p, dd in json.loads(f.read_text())["detections"].items()}


def wilcoxon_table(runs, base, arm, grid="int"):
    """Paired two-sided Wilcoxon (exact for few runs) of ``arm - base`` over the runs, per metric."""
    from scipy.stats import wilcoxon

    out = {}
    for k in WILCOXON_KEYS:
        a = np.array([r[arm][grid][k] for r in runs.values()])
        b = np.array([r[base][grid][k] for r in runs.values()])
        d = a - b
        p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
        out[k] = {"base": float(b.mean()), "arm": float(a.mean()), "delta": float(d.mean()), "delta_sd": float(d.std()),
                  "p": p, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d)}
    return out


def decision(w):
    """Rule fixed before the CV: tight +0.03, mAP@1 up with p < 0.05, mAP@5 not down by more than 0.01."""
    tight = w["avg_mAP_tight"]["delta"] >= 0.03
    map1 = w["mAP@1"]["delta"] > 0 and w["mAP@1"]["p"] < 0.05
    map5 = w["mAP@5"]["delta"] >= -0.01
    return {"tight_up_0.03": tight, "mAP@1_up_p<0.05": map1, "mAP@5_drop<=0.01": map5, "pass": tight and map1 and map5}


def cv_main(args):
    from train_refiner import cc_detections

    report = {"arm": args.cv, "folds": args.folds, "seeds": args.seeds, "runs": {}, "n_events": {}}
    comparisons = [("R0", args.cv)] + ([("R0_cc", f"{args.cv}_cc")] if args.cc else [])
    for fold in args.folds:
        fi, split, bg, names = fold_meta(f"{fold}/5")
        ev = load_event_times(split["val"], names)
        report["n_events"][f"fold{fi}"] = sum(len(v) for v in ev.values())
        gts = {"int": {p: [(c, float(s)) for c, s, _ in ev[p]] for p in ev},
               "frac": {p: [(c, t) for c, _, t in ev[p]] for p in ev}}
        for seed in args.seeds:
            direct, lengths, _ = fbase_detections(seed, fi, bg)
            lengths = {p: float(n) for p, n in lengths.items()}
            run = {}
            for base, arm in comparisons:
                for a in (base, arm):
                    run[a] = {g: metrics(load_dets(a, seed, fi), gts[g], lengths, len(names), bg) for g in ("int", "frac")}
            assert run["R0"]["int"] == metrics(direct, gts["int"], lengths, len(names), bg), "R0 != F_base"
            if args.cc:
                cc, _, prep = cc_detections(seed, fi)
                assert all(sorted(prep.gts[p]) == sorted(gts["int"][p]) for p in gts["int"]), "cc ground truth differs"
                assert run["R0_cc"]["int"] == metrics(cc, gts["int"], lengths, len(names), bg), "R0_cc != caps+chain"
                ref = json.loads((ROOT / "EXP2/F_base" / f"seed{seed}" / f"fold{fi}" / "val" / "constraints_nms5.json")
                                 .read_text())["results"]["caps_chain|a=0.1|s=1|f=0.2"]["avg_map_tight"]
                run["R0_cc"]["eval_constraints_avg_map_tight"] = ref
            report["runs"][f"seed{seed}/fold{fi}"] = run
            print(f"seed{seed} fold{fi}: " + " | ".join(
                f"{a} @1 {run[a]['int']['mAP@1']:.3f} tight {run[a]['int']['avg_mAP_tight']:.3f} "
                f"@5 {run[a]['int']['mAP@5']:.3f} F1 {run[a]['int']['F1@5']:.3f}" for a in run), flush=True)
    runs = report["runs"]
    arms = [a for pair in comparisons for a in pair]
    report["mean"] = {a: {g: {k: float(np.mean([r[a][g][k] for r in runs.values()])) for k in runs[next(iter(runs))][a][g]}
                          for g in ("int", "frac")} for a in arms}
    report["wilcoxon"], report["decision"] = {}, {}
    for base, arm in comparisons:
        for g in ("int", "frac"):
            report["wilcoxon"][f"{arm}_vs_{base}/{g}"] = wilcoxon_table(runs, base, arm, g)
        report["decision"][f"{arm}_vs_{base}"] = decision(report["wilcoxon"][f"{arm}_vs_{base}/int"])
    print(json.dumps({"wilcoxon_int": {k: v for k, v in report["wilcoxon"].items() if k.endswith("/int")},
                      "decision": report["decision"]}, indent=1))
    (EXP / f"cv_{args.cv}.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", default="0/5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=["R0", "R1-1fps", "R1"])
    ap.add_argument("--cv", help="arm compared with R0 over every --folds x --seeds run")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cc", action="store_true", help="also the caps + chain detections (--cv)")
    args = ap.parse_args()
    if args.cv:
        return cv_main(args)
    fi, split, bg, names = fold_meta(args.fold)
    ev = load_event_times(split["val"], names)
    gts = {"int": {p: [(c, float(s)) for c, s, _ in ev[p]] for p in ev},
           "frac": {p: [(c, t) for c, _, t in ev[p]] for p in ev}}
    report = {"fold": args.fold, "val_patients": split["val"], "n_events": sum(len(v) for v in ev.values()),
              "runs": {}}
    for seed in args.seeds:
        direct, lengths, _ = fbase_detections(seed, fi, bg)
        rowsec = {}
        for p, dd in direct.items():
            sec = np.load(ROOT / "EXP2/F_base" / f"seed{seed}" / f"fold{fi}" / "val" / f"{p}.npz")["second"]
            rowsec[p] = [(c, float(sec[t]), s) for c, t, s in dd]
        lengths = {p: float(n) for p, n in lengths.items()}
        arms = {}
        for arm in args.arms:
            f = EXP / arm / f"seed{seed}" / f"fold{fi}" / "val" / "detections.json"
            if f.exists():
                arms[arm] = {p: [tuple(d) for d in dd] for p, dd in json.loads(f.read_text())["detections"].items()}
        arms["R0_direct"] = direct
        arms["R0_rowsec"] = rowsec
        arms["oracle_4fps"] = oracle(direct, direct, gts["frac"], 4)
        arms["oracle_1fps"] = oracle(direct, direct, gts["frac"], 1)
        run = {}
        for arm, dets in arms.items():
            run[arm] = {g: metrics(dets, gts[g], lengths, len(names), bg) for g in ("int", "frac")}
        if "R0" in run:
            assert run["R0"] == run["R0_direct"], "R0 (identity refiner) does not reproduce F_base"
        report["runs"][f"seed{seed}"] = run
        for arm, m in run.items():
            print(f"seed{seed} {arm:12s} int: mAP@1 {m['int']['mAP@1']:.3f} @2 {m['int']['mAP@2']:.3f} "
                  f"@5 {m['int']['mAP@5']:.3f} tight {m['int']['avg_mAP_tight']:.3f} F1 {m['int']['F1@5']:.3f} "
                  f"err<=1 {m['int']['err<=1']:.2f} | frac: @1 {m['frac']['mAP@1']:.3f} @2 {m['frac']['mAP@2']:.3f} "
                  f"err<=1 {m['frac']['err<=1']:.2f}", flush=True)
    runs = report["runs"]
    present = [a for a in ("R0", "R1-1fps", "R1") if all(a in r for r in runs.values())]
    report["mean"] = {a: {g: {k: float(np.mean([r[a][g][k] for r in runs.values()])) for k in runs["seed0"][a][g]}
                          for g in ("int", "frac")} for a in runs["seed0"] if all(a in r for r in runs.values())}
    if {"R0", "R1", "R1-1fps"} <= set(present):
        d1 = {s: r["R1"]["int"]["mAP@1"] - r["R0"]["int"]["mAP@1"] for s, r in runs.items()}
        beats = {s: r["R1"]["int"]["mAP@1"] > r["R1-1fps"]["int"]["mAP@1"] for s, r in runs.items()}
        report["gate"] = {"delta_mAP@1_R1_vs_R0": d1, "R1_beats_R1-1fps": beats,
                          "pass": all(v >= 0.03 for v in d1.values()) and all(beats.values())}
        print(json.dumps(report["gate"], indent=1))
    (EXP / f"pilot_fold{fi}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
