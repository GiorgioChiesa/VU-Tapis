#!/usr/bin/env python3
"""
Experiment 7 table: phase-only temporal models (A1, A2, A3) against F_base.

Every arm is read from the shared posterior format (``outputs/orsi/EXP7/<arm>/seed<s>/fold<i>/<sub>/``,
see tools/phase_metrics.py) and evaluated with the same code, F_base included:
  --export-fbase   writes EXP7/F_base/seed*/fold*/phase_posteriors from the ``probs_phases`` saved by
                   train_hybrid.py in EXP2/F_base/seed*/fold*/val (test fold of each run)
Variants: ``<arm>`` = selected epoch (phase_posteriors), ``<arm>@last`` = last epoch
(phase_posteriors_last), and any other ``phase_posteriors*`` sub-directory (e.g. Esp6 decoders) as
``<arm>@<suffix>``.

  --import-decoded DIR   stores the labels decoded by Esp6 ``tools/exp6_decode_cv.py`` (``DIR/<arm>/labels/<arm>/
                   seed<s>/<variant>/<video>.npy``, run on the per-seed merged posteriors) as one-hot posteriors in
                   ``<arm>/seed<s>/fold<k>/phase_posteriors_<suffix>``, so they appear as ``<arm>@<suffix>``

Paired over the 10 (seed, fold) runs against F_base: mean difference, Wilcoxon p, wins. Pre-registered rule:
an arm is adopted if seg F1@50 >= F_base + 5 points in >= 8/10 runs and the mean frame macro-F1 over the
runs >= F_base - 0.01. Aggregate output only: summary.json and a markdown table on stdout.

Video level (``videolevel`` in summary.json): the two seeds share the same outer folds, so the 10 runs are not
independent units. Per variant and video the per-video metric is averaged over the seeds; paired two-sided
Wilcoxon over the 50 videos and over the 5 outer folds (fold mean of the seed-averaged video metric) against
F_base and against F_base@P1 (when present). Pooled metrics are also given over all 50 videos of a seed
(as Esp6), mean over seeds.

Example:
    python tools/summarize_exp7.py --export-fbase
    python tools/summarize_exp7.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import wilcoxon

import phase_metrics as pm

ROOT = Path(__file__).resolve().parents[1] / "outputs/orsi"
EXP = ROOT / "EXP7"
REF = "F_base"
KEYS = ["F1@50", "F1@25", "F1@10", "edit", "macro_F1", "acc", "jaccard", "boundary_F1@5s", "seg_ratio",
        "pooled_F1@50", "pooled_edit", "pooled_changes_ratio", "pooled_macro_F1", "pooled_acc"]
RULE_DELTA, RULE_MIN_RUNS, RULE_MACRO_TOL = 5.0, 8, 0.01


def export_fbase(frame_dir):
    from train_hybrid import load_orsi

    src = ROOT / "EXP2/F_base"
    feat = ROOT / "LONG/cv_ft/fold0/untrimmed/checkpoint_best_steps/all"  # labels only; any fold's features
    ns = argparse.Namespace(feat_dir=str(feat), frame_dir=frame_dir, cv_fold=None, target="point", k=3, sigma=2.0,
                            streams="dynamic")
    tmp = EXP / "_labels"
    tmp.mkdir(parents=True, exist_ok=True)
    data, _, info = load_orsi(ns, tmp)
    for val in sorted(src.glob("seed*/fold*/val")):
        run = val.parent
        fold = int(run.name.removeprefix("fold"))
        meta = json.loads((val / "meta.json").read_text())
        assert meta["cv_fold"] == f"{fold}/5", (run, meta["cv_fold"])
        probs, gts = {}, {}
        for z in sorted(val.glob("RARP*.npz")):
            p = np.load(z)["probs_phases"]
            assert len(p) == len(data[z.stem]["phase"]), z
            probs[z.stem], gts[z.stem] = p, data[z.stem]["phase"]
        d = pm.write_posteriors(EXP / REF / run.parent.name / run.name, probs, gts, fold, info["ph_names"])
        (d.parent / "classes.json").write_text(json.dumps(info["ph_names"]))
        print(f"{d}: {len(probs)} videos", flush=True)


def import_decoded(dec_root, variant, suffix):
    for arm_dir in sorted(Path(dec_root).iterdir()):
        lab_root = arm_dir / "labels" / arm_dir.name
        if not lab_root.is_dir():
            continue
        arm = arm_dir.name
        for seed_dir in sorted(lab_root.glob("seed*")):
            src = seed_dir / variant
            n = 0
            for run in sorted((EXP / arm / seed_dir.name).glob("fold*")):
                probs, gts, folds = pm.read_posteriors(run / "phase_posteriors")
                classes = json.loads((run / "phase_posteriors/classes.json").read_text())
                onehot = {}
                for v, p in probs.items():
                    lab = np.load(src / f"{v}.npy").astype(int)
                    assert len(lab) == len(p), (arm, v)
                    onehot[v] = np.eye(p.shape[1], dtype=np.float32)[lab]
                n += len(onehot)
                fold = int(run.name.removeprefix("fold"))
                pm.write_posteriors(run, onehot, gts, fold, classes, sub=f"phase_posteriors_{suffix}")
            print(f"{arm}/{seed_dir.name}@{suffix}: {n} videos from {src}", flush=True)


VL_KEYS = ["F1@50", "F1@25", "F1@10", "edit", "macro_F1", "acc", "jaccard", "boundary_F1@5s", "seg_ratio"]


def _wx(x, r):
    return float(wilcoxon(x, r).pvalue) if np.any(np.asarray(x) != np.asarray(r)) else 1.0


def videolevel():
    """Seed-averaged per-video metrics, paired Wilcoxon over videos (n=50) and folds (n=5)."""
    from tapis.evaluate import segment_eval
    per, pooled, fold_of = {}, {}, {}
    for d in sorted(EXP.glob("*/seed*/fold*/phase_posteriors*")):
        if not d.is_dir() or d.parts[-4].startswith("_") or not list(d.glob("*.npz")):
            continue
        arm, seed = d.parts[-4], d.parts[-3]
        suffix = d.name.removeprefix("phase_posteriors").lstrip("_")
        var = f"{arm}@{suffix}" if suffix else arm
        probs, gts, folds = pm.read_posteriors(d)
        for v in probs:
            lab = probs[v].argmax(1)
            per.setdefault(var, {}).setdefault(v, {})[seed] = pm.video_metrics(lab, gts[v])
            assert fold_of.setdefault(v, folds[v]) == folds[v]
            pooled.setdefault(var, {}).setdefault(seed, ({}, {}))[0][v] = lab
            pooled[var][seed][1][v] = gts[v]
    vids = sorted(fold_of)
    fo = np.array([fold_of[v] for v in vids])
    fids = sorted(set(fo.tolist()))
    mat = {}
    for var, pv in per.items():
        if sorted(pv) != vids or any(len(s) != 2 for s in pv.values()):
            print(f"videolevel: skip {var} (incomplete)")
            continue
        mat[var] = {k: np.array([np.mean([m[k] for m in pv[v].values()]) for v in vids]) for k in VL_KEYS}
    refs = [r for r in (REF, f"{REF}@P1") if r in mat]
    out = {"n_videos": len(vids), "folds": fids, "references": refs, "variants": {}}
    for var in mat:
        e = {k: {"mean": float(mat[var][k].mean()), "median": float(np.median(mat[var][k]))} for k in VL_KEYS}
        pl = [segment_eval.segmental_metrics(*pooled[var][s]) for s in sorted(pooled[var])]
        e["pooled_all_videos"] = {k: float(np.mean([p[k] for p in pl])) for k in ("F1@50", "edit")}
        for r in refs:
            if r == var:
                continue
            e[f"vs_{r}"] = {}
            for k in VL_KEYS:
                x, y = mat[var][k], mat[r][k]
                xf = np.array([x[fo == f].mean() for f in fids])
                yf = np.array([y[fo == f].mean() for f in fids])
                e[f"vs_{r}"][k] = {"delta_mean": float((x - y).mean()), "n_improved": int((x > y).sum()),
                                   "n_worse": int((x < y).sum()), "p_videos": _wx(x, y),
                                   "fold_n_improved": int((xf > yf).sum()), "p_folds": _wx(xf, yf)}
        out["variants"][var] = e
    return out


def collect():
    """{variant: {(seed, fold): metrics}} for every posterior directory under EXP7."""
    out = {}
    for d in sorted(EXP.glob("*/seed*/fold*/phase_posteriors*")):
        if not d.is_dir() or d.parts[-4].startswith("_"):
            continue
        arm = d.parts[-4]
        suffix = d.name.removeprefix("phase_posteriors").lstrip("_")
        variant = f"{arm}@{suffix}" if suffix else arm
        key = (d.parts[-3], d.parts[-2])
        if not list(d.glob("*.npz")):
            continue
        out.setdefault(variant, {})[key] = pm.posterior_metrics(d)
    return out


def paired(runs, ref_runs):
    keys = sorted(set(runs) & set(ref_runs))
    res = {"n_paired": len(keys)}
    for k in KEYS:
        a = np.array([runs[r][k] for r in keys])
        b = np.array([ref_runs[r][k] for r in keys])
        diff = a - b
        p = float(wilcoxon(a, b).pvalue) if len(keys) > 1 and np.any(diff) else 1.0
        res[k] = {"diff_mean": float(diff.mean()), "diff_std": float(diff.std()), "p": p,
                  "wins": int((diff > 0).sum())}
    f1 = np.array([runs[r]["F1@50"] - ref_runs[r]["F1@50"] for r in keys])
    macro = float(np.mean([runs[r]["macro_F1"] for r in keys]) - np.mean([ref_runs[r]["macro_F1"] for r in keys]))
    n_ok = int((f1 >= RULE_DELTA).sum())
    res["rule"] = {"runs_F1@50_ge_ref+5": n_ok, "macro_F1_diff": macro,
                   "adopted": bool(len(keys) == 10 and n_ok >= RULE_MIN_RUNS and macro >= -RULE_MACRO_TOL)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-fbase", action="store_true")
    ap.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--import-decoded", metavar="DIR")
    ap.add_argument("--decoded-variant", default="P1_default")
    ap.add_argument("--decoded-suffix", default="P1")
    args = ap.parse_args()
    if args.import_decoded:
        import_decoded(args.import_decoded, args.decoded_variant, args.decoded_suffix)
        return
    if args.export_fbase:
        export_fbase(args.frame_dir)
        return
    runs = collect()
    classes = json.loads(next(EXP.glob(f"{REF}/seed*/fold*/phase_posteriors/classes.json")).read_text())
    table = {}
    for v, rr in runs.items():
        table[v] = {"n_runs": len(rr), **{k: [float(np.mean([m[k] for m in rr.values()])),
                                              float(np.std([m[k] for m in rr.values()]))] for k in KEYS}}
        table[v]["per_phase_F1"] = np.nanmean([m["per_phase_F1"] for m in rr.values()], 0).tolist()
        table[v]["runs"] = {f"{s}/{f}": {k: m[k] for k in KEYS} for (s, f), m in sorted(rr.items())}
    vs_ref = {v: paired(rr, runs[REF]) for v, rr in runs.items() if v != REF and REF in runs}
    times = {}
    for arm in sorted({v.split("@")[0] for v in runs} - {REF}):
        t = [json.loads(f.read_text()) for f in EXP.glob(f"{arm}/seed*/fold*/metrics.json")]
        if t:
            times[arm] = {"time_s_mean": float(np.mean([x["time_s"] for x in t])), "n_params": t[0]["n_params"],
                          "e_star": [x["selected"]["epoch"] for x in t]}
    vl = videolevel()
    (EXP / "summary.json").write_text(json.dumps(
        {"classes": classes, "table": table, "vs_F_base": vs_ref, "train": times, "videolevel": vl,
         "rule": f"F1@50 >= F_base + {RULE_DELTA} in >= {RULE_MIN_RUNS}/10 runs and mean macro-F1 >= F_base - "
                 f"{RULE_MACRO_TOL}"}, indent=1))
    cols = ["F1@50", "F1@25", "F1@10", "edit", "macro_F1", "acc", "jaccard", "boundary_F1@5s", "seg_ratio",
            "pooled_F1@50", "pooled_changes_ratio"]
    print("| arm | n | " + " | ".join(cols) + " | runs >= +5 | adopted |")
    print("|" + "---|" * (len(cols) + 4))
    for v in [REF] + sorted(set(table) - {REF}):
        r = table[v]
        cells = [f"{r[k][0]:.3f}" if r[k][0] < 10 else f"{r[k][0]:.1f}" for k in cols]
        rule = vs_ref.get(v, {}).get("rule", {})
        print(f"| {v} | {r['n_runs']} | " + " | ".join(cells)
              + f" | {rule.get('runs_F1@50_ge_ref+5', '-')} | {rule.get('adopted', '-')} |")
    for v, res in vs_ref.items():
        print(f"{v} vs {REF}: " + ", ".join(f"{k} {res[k]['diff_mean']:+.3f} (p {res[k]['p']:.3f}, {res[k]['wins']}/"
                                             f"{res['n_paired']})" for k in ("F1@50", "edit", "macro_F1", "acc")))

    print("\nvideo level (seed-averaged, n = %d videos / %d folds)" % (vl["n_videos"], len(vl["folds"])))
    for v, e in vl["variants"].items():
        for r in vl["references"]:
            if f"vs_{r}" in e:
                c = e[f"vs_{r}"]["F1@50"]
                print(f"{v} vs {r}: F1@50 {c['delta_mean']:+.1f}, videos +{c['n_improved']}/-{c['n_worse']}, "
                      f"p {c['p_videos']:.2g}; folds {c['fold_n_improved']}/5, p {c['p_folds']:.3g}; "
                      f"pooled(50 videos) F1@50 {e['pooled_all_videos']['F1@50']:.1f}")


if __name__ == "__main__":
    main()
