#!/usr/bin/env python3
"""
Esp6b: cross-validated comparison of phase decoders (tools/phase_decode.py) on saved posteriors.

Every ``--runs`` directory holds ``phase_posteriors/`` in the shared format (``<video>.npz`` with
``probs`` [T, C], ``gt`` [T], ``fold``; ``classes.json``). For each run and each outer fold k:

  * test videos  = fold k; their priors (durations, transitions, order) come from the other folds;
  * tuned variants pick their hyperparameters on the training videos only: each training video of
    inner fold j is decoded from its own out-of-fold posteriors with priors fitted on the folds
    outside {k, j}; objective = mean per-video F1@50 (grid order breaks ties: default first);
  * ``*_default`` variants use fixed parameters, no tuning.

A run x fold is one paired unit (5 folds x 2 seeds = 10). ``summary.json`` / ``summary.md`` give per
variant the mean over units of the per-unit metrics (mean over the test videos), the paired
differences vs ``P0_argmax``, two-sided Wilcoxon p, the pre-registered decision rule, and per-phase
frame F1 / segmental F1@50 (pooled within a unit, averaged over units). Metrics are given with Idle
(``all``) and without Idle (``noidle``: Idle seconds and segments dropped).

Usage (CPU, OMP_NUM_THREADS=1, one process per worker):
    python tools/exp6_decode_cv.py --runs outputs/orsi/EXP6/F_base/seed0 outputs/orsi/EXP6/F_base/seed1 \\
        --out outputs/orsi/EXP6/decode --workers 8
    python tools/exp6_decode_cv.py ... --summarize-only        # rebuild the summary from units/*.json
    python tools/exp6_decode_cv.py ... --videolevel            # video-level stats from labels/ (no decoding)
Outputs stay on the server: ``units/<run>__fold<k>__<variant>.json`` (aggregates and chosen
parameters) and ``labels/<run>/<variant>/<video>.npy`` (decoded test labels).
"""

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase_decode as pd_  # noqa: E402
from tapis.evaluate.segment_eval import segmental_metrics  # noqa: E402

REF = "P0_argmax"
GRIDS = {
    "moving_avg": {"window": [15, 5, 9, 31, 61, 121]},
    "hysteresis": {"margin": [0.1, 0.0, 0.05, 0.2, 0.3, 0.5], "q": [5.0, 10.0, 25.0], "factor": [0.5, 0.25, 1.0]},
    "hsmm": {"temp": [1.0, 2.0, 4.0], "dur_weight": [1.0, 0.5, 2.0], "trans_weight": [1.0, 0.5, 2.0]},
    "pelt": {"penalty": [10.0, 2.0, 5.0, 20.0, 50.0], "min_size": [10, 5, 30]},
}
PP_MARGINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
PP_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
# name -> (decoder, fixed params, tuning: None | "grid" | "perphase")
VARIANTS = {
    "P0_argmax": ("argmax", {}, None),
    "P0_ma15": ("moving_avg", {"window": 15}, None),
    "P0_ma_tuned": ("moving_avg", {}, "grid"),
    "P1_default": ("hysteresis", {}, None),
    "P1_tuned": ("hysteresis", {}, "grid"),
    "P1_perphase": ("hysteresis", {}, "perphase"),
    "P2_learned_default": ("hsmm", {"mode": "learned"}, None),
    "P2_learned_tuned": ("hsmm", {"mode": "learned"}, "grid"),
    "P2_order_default": ("hsmm", {"mode": "order"}, None),
    "P2_order_tuned": ("hsmm", {"mode": "order"}, "grid"),
    "P2_free_default": ("hsmm", {"mode": "free"}, None),
    "P2_free_tuned": ("hsmm", {"mode": "free"}, "grid"),
    "P3_default": ("pelt", {}, None),
    "P3_tuned": ("pelt", {}, "grid"),
}
RULE = {"min_gain": 10.0, "min_runs": 8, "macro_F1_drop": 0.01, "acc_drop": 0.005, "ratio": (0.8, 1.3)}
_CACHE = {}


def load(run):
    if run not in _CACHE:
        _CACHE[run] = pd_.load_posteriors(Path(run) / "phase_posteriors")
    return _CACHE[run]


def f1_50(pred, gt):
    keep = gt >= 0
    tp, fp, fn, _ = pd_._seg_counts(pred[keep], gt[keep], 0.5)
    return 100 * pd_._f1(tp, fp, fn)


def objective(dec, params, inner):
    """Mean per-video F1@50 over ``inner`` = [(probs, gt, priors)]."""
    return float(np.mean([f1_50(pd_.decode(dec, p, pr, **params), g) for p, g, pr in inner]))


def tune_grid(dec, fixed, grid, inner):
    keys = list(grid)
    best, best_p, n = -1.0, None, 0
    for vals in itertools.product(*(grid[k] for k in keys)):
        p = {**fixed, **dict(zip(keys, vals))}
        s = objective(dec, p, inner)
        n += 1
        if s > best:
            best, best_p = s, p
    return best_p, best, n


def tune_perphase(fixed, inner, C, order, passes=2):
    """P1 with one hysteresis margin and one minimum-duration scale per phase, by coordinate descent
    (phases in modal order) from the best global setting."""
    p0, s0, n = tune_grid("hysteresis", fixed, GRIDS["hysteresis"], inner)
    margin = np.full(C, p0["margin"])
    scale = np.ones(C)
    best = s0
    for _ in range(passes):
        changed = False
        for k in order:
            for m, sc in itertools.product(PP_MARGINS, PP_SCALES):
                if m == margin[k] and sc == scale[k]:
                    continue
                mg, sl = margin.copy(), scale.copy()
                mg[k], sl[k] = m, sc
                s = objective("hysteresis", {"margin": mg, "q": p0["q"], "factor": p0["factor"] * sl}, inner)
                n += 1
                if s > best + 1e-9:
                    best, margin, scale, changed = s, mg, sl, True
        if not changed:
            break
    params = {"margin": margin.tolist(), "q": p0["q"], "factor": (p0["factor"] * scale).tolist()}
    return params, best, n


def run_unit(task):
    run, name, k, variant, out = task
    t0 = time.time()
    videos, names = load(run)
    C = len(names)
    dec, fixed, tuning = VARIANTS[variant]
    train = {v: d for v, d in videos.items() if d["fold"] != k}
    test = {v: d for v, d in videos.items() if d["fold"] == k}
    pri_test = pd_.fit_priors([d["gt"] for d in train.values()], C)
    params, obj, n_eval = dict(fixed), None, 0
    if tuning:
        inner_pri = {j: pd_.fit_priors([d["gt"] for d in train.values() if d["fold"] != j], C)
                     for j in {d["fold"] for d in train.values()}}
        inner = [(d["probs"], d["gt"], inner_pri[d["fold"]]) for d in train.values()]
        if tuning == "grid":
            params, obj, n_eval = tune_grid(dec, fixed, GRIDS[dec], inner)
        else:
            params, obj, n_eval = tune_perphase(fixed, inner, C, pri_test.order)
    ldir = Path(out) / "labels" / name / variant
    ldir.mkdir(parents=True, exist_ok=True)
    per = {"all": [], "noidle": []}
    preds, gts = {}, {}
    for v, d in sorted(test.items()):
        lab = pd_.decode(dec, d["probs"], pri_test, **params)
        np.save(ldir / f"{v}.npy", lab.astype(np.int16))
        per["all"].append(pd_.video_metrics(lab, d["gt"], C))
        per["noidle"].append(pd_.video_metrics(lab, d["gt"], C, ignore_idle=True))
        keep = d["gt"] >= 0
        preds[v], gts[v] = lab[keep], d["gt"][keep]
    res = {"run": name, "fold": k, "variant": variant, "decoder": dec, "tuning": tuning, "classes": names,
           "params": {a: (b.tolist() if isinstance(b, np.ndarray) else b) for a, b in params.items()},
           "tune_objective_F1@50": obj, "n_evals": n_eval, "n_test": len(test), "n_train": len(train),
           **{s: pd_.aggregate(m, C) for s, m in per.items()}, "seconds": time.time() - t0}
    # cross-check of the pooled segmental metrics against tapis/evaluate/segment_eval.py
    ref = segmental_metrics(preds, gts)
    for kk in (10, 25, 50):
        assert abs(ref[f"F1@{kk}"] - res["all"][f"pooled_F1@{kk}"]) < 1e-6, (name, k, variant, kk)
    f = Path(out) / "units" / f"{name.replace('/', '_')}__fold{k}__{variant}.json"
    f.write_text(json.dumps(res, indent=1))
    return f.name, round(res["seconds"]), res["all"]["F1@50"]


def summarize(out, variants):
    units = [json.loads(f.read_text()) for f in sorted((Path(out) / "units").glob("*.json"))]
    by = {}
    for u in units:
        by.setdefault(u["variant"], {})[(u["run"], u["fold"])] = u
    keys = sorted(by[REF])
    metrics = pd_.PUBLIC + ("pooled_F1@50",)
    summary = {"units": [f"{r}/fold{k}" for r, k in keys], "classes": units[0]["classes"], "rule": RULE,
               "variants": {}}
    for name in [v for v in variants if v in by]:
        if sorted(by[name]) != keys:
            print(f"skip {name}: {len(by[name])}/{len(keys)} units")
            continue
        entry = {"params": [by[name][u]["params"] for u in keys]}
        for scope in ("all", "noidle"):
            sc = {}
            for m in metrics:
                x = np.array([by[name][u][scope][m] for u in keys])
                r = np.array([by[REF][u][scope][m] for u in keys])
                d = x - r
                p = float(wilcoxon(x, r).pvalue) if np.any(d != 0) else 1.0
                sc[m] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "delta": float(d.mean()),
                         "delta_min": float(d.min()), "delta_max": float(d.max()), "wilcoxon_p": p,
                         "per_unit": x.tolist()}
            pc = [by[name][u][scope]["per_class"] for u in keys]
            sc["per_class"] = {
                key: [None if all(p_[key][c] is None for p_ in pc) else
                      float(np.mean([p_[key][c] for p_ in pc if p_[key][c] is not None]))
                      for c in range(len(pc[0][key]))]
                for key in ("frame_F1", "jaccard", "seg_F1@50")}
            sc["confusion"] = np.sum([by[name][u][scope]["confusion"] for u in keys], 0).tolist()
            entry[scope] = sc
        a = entry["all"]
        gains = np.array(a["F1@50"]["per_unit"]) - np.array([by[REF][u]["all"]["F1@50"] for u in keys])
        checks = {"F1@50_gain_runs": int((gains >= RULE["min_gain"]).sum()),
                  "macro_F1_delta": a["macro_F1"]["delta"], "acc_delta": a["acc"]["delta"],
                  "seg_ratio": a["seg_ratio"]["mean"]}
        checks["wins"] = bool(checks["F1@50_gain_runs"] >= RULE["min_runs"]
                              and checks["macro_F1_delta"] >= -RULE["macro_F1_drop"]
                              and checks["acc_delta"] >= -RULE["acc_drop"]
                              and RULE["ratio"][0] <= checks["seg_ratio"] <= RULE["ratio"][1])
        entry["rule"] = checks if name != REF else None
        summary["variants"][name] = entry
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=1))
    (Path(out) / "summary.md").write_text(markdown(summary))
    print(markdown(summary))


def markdown(s):
    cols = [("F1@50", 1), ("F1@25", 1), ("F1@10", 1), ("edit", 1), ("macro_F1", 3), ("acc", 3),
            ("jaccard", 3), ("boundary_F1", 3), ("seg_ratio", 2)]
    lines = [f"Units: {len(s['units'])} (run x outer fold). Mean over units of the per-video mean; "
             "delta vs P0_argmax in brackets; * Wilcoxon p < 0.05.", ""]
    for scope in ("all", "noidle"):
        lines += [f"### {'with Idle' if scope == 'all' else 'without Idle'}", "",
                  "| variant | " + " | ".join(c for c, _ in cols) + (" | runs +10 | rule |" if scope == "all" else " |"),
                  "|---" * (len(cols) + 1 + (2 if scope == "all" else 0)) + "|"]
        for name, e in s["variants"].items():
            cells = []
            for c, nd in cols:
                m = e[scope][c]
                star = "*" if name != REF and m["wilcoxon_p"] < 0.05 else ""
                cells.append(f"{m['mean']:.{nd}f}" + ("" if name == REF else f" ({m['delta']:+.{nd}f}{star})"))
            row = f"| {name} | " + " | ".join(cells) + " |"
            if scope == "all":
                r = e["rule"]
                row += (" - | - |" if r is None else f" {r['F1@50_gain_runs']}/{len(s['units'])} | {'WIN' if r['wins'] else 'fail'} |")
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


VL_METRICS = ("F1@50", "F1@25", "F1@10", "edit", "acc", "macro_F1", "boundary_F1")
VL_POOLED = ("pooled_F1@50", "pooled_F1@25", "pooled_F1@10", "pooled_acc", "pooled_macro_F1", "pooled_boundary_F1")
VL_MAIN = ("P0_argmax", "P1_default", "P2_free_tuned", "P3_tuned")


def run_names(runs):
    """``[(run dir, run name)]`` with the names used for ``units/`` and ``labels/`` (as in ``main``)."""
    common = os.path.commonpath([str(Path(r).resolve()) for r in runs])
    return [(r, os.path.relpath(Path(r).resolve(), Path(common).parent) if len(runs) > 1 else Path(r).name)
            for r in runs]


def _wilcoxon(x, r):
    d = np.asarray(x) - np.asarray(r)
    return float(wilcoxon(x, r).pvalue) if np.any(d != 0) else 1.0


def summarize_videolevel(runs, out, variants):
    """
    Video-level re-analysis from the saved decoded labels (``labels/<run>/<variant>/<video>.npy``), no
    re-decoding. The runs (seeds) share the same outer folds, so the run x fold units of ``summary.json``
    are not independent. Here, per variant and video, the per-video metric is averaged over the runs;
    paired two-sided Wilcoxon vs ``P0_argmax`` over the videos (n = 50) and over the outer folds (n = 5,
    per fold the mean of the run-averaged video metrics). Also pooled metrics (counts summed over all
    videos of a run, as ``tapis/evaluate/segment_eval.py`` and MS-TCN), averaged over the runs.
    Writes ``summary_videolevel.{json,md}``.
    """
    out = Path(out)
    named = run_names(runs)
    gts, folds, names = {}, {}, None
    for run, name in named:
        videos, names = load(run)
        for v, d in videos.items():
            assert folds.setdefault(v, d["fold"]) == d["fold"], "a video has different folds across runs"
        gts[name] = {v: d["gt"] for v, d in videos.items()}
    C = len(names)
    vids = sorted(folds)
    fold_ids = sorted(set(folds.values()))
    fold_of = np.array([folds[v] for v in vids])
    units = {}
    for f in (out / "units").glob("*.json"):
        u = json.loads(f.read_text())
        units[(u["run"], u["fold"], u["variant"])] = u
    per, pooled, check = {}, {}, 0.0
    for var in [v for v in variants if v in VARIANTS]:
        if not all((out / "labels" / name / var).is_dir() for _, name in named):
            print(f"skip {var}: labels missing")
            continue
        per[var] = {sc: {m: np.zeros((len(named), len(vids))) for m in VL_METRICS} for sc in ("all", "noidle")}
        pooled[var] = {}
        for i, (_, name) in enumerate(named):
            mv = {"all": [], "noidle": []}
            for j, v in enumerate(vids):
                lab = np.load(out / "labels" / name / var / f"{v}.npy").astype(int)
                for sc in ("all", "noidle"):
                    m = pd_.video_metrics(lab, gts[name][v], C, ignore_idle=sc == "noidle")
                    mv[sc].append(m)
                    for k in VL_METRICS:
                        per[var][sc][k][i, j] = m[k]
            pooled[var][name] = {sc: {k: pd_.aggregate(mv[sc], C)[k] for k in VL_POOLED} for sc in ("all", "noidle")}
            # cross-check: per-fold mean of the recomputed per-video F1@50 = saved unit value
            for k in fold_ids:
                u = units.get((name, k, var))
                if u is not None:
                    check = max(check, abs(per[var]["all"]["F1@50"][i, fold_of == k].mean() - u["all"]["F1@50"]))
    assert check < 1e-6, f"recomputed metrics differ from units/: {check}"
    res = {"videos": len(vids), "runs": [n for _, n in named], "folds": fold_ids, "reference": REF,
           "max_abs_diff_vs_units_F1@50": check, "variants": {}}
    for var in per:
        e = {}
        for sc in ("all", "noidle"):
            e[sc] = {}
            for k in VL_METRICS:
                x = per[var][sc][k].mean(0)          # run-averaged, one value per video
                r = per[REF][sc][k].mean(0)
                d = x - r
                xf = np.array([x[fold_of == f].mean() for f in fold_ids])
                rf = np.array([r[fold_of == f].mean() for f in fold_ids])
                e[sc][k] = {"mean": float(x.mean()), "median": float(np.median(x)),
                            "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
                            "n_improved": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
                            "n_tied": int((d == 0).sum()), "wilcoxon_p_videos": _wilcoxon(x, r),
                            "fold_delta_mean": float((xf - rf).mean()),
                            "fold_n_improved": int((xf > rf).sum()), "wilcoxon_p_folds": _wilcoxon(xf, rf),
                            "per_video_sorted_ids": x.tolist()}
            e[sc]["pooled"] = {k: {"mean_runs": float(np.mean([pooled[var][n][sc][k] for _, n in named])),
                                   "per_run": [pooled[var][n][sc][k] for _, n in named],
                                   "delta": float(np.mean([pooled[var][n][sc][k] - pooled[REF][n][sc][k]
                                                           for _, n in named]))} for k in VL_POOLED}
            us = [units[(n, f, var)][sc]["pooled_F1@50"] for _, n in named for f in fold_ids if (n, f, var) in units]
            e[sc]["pooled_within_unit_F1@50_mean"] = float(np.mean(us)) if us else None
        res["variants"][var] = e
    (out / "summary_videolevel.json").write_text(json.dumps(res, indent=1))
    (out / "summary_videolevel.md").write_text(markdown_videolevel(res))
    print(markdown_videolevel(res, main_only=True))


def markdown_videolevel(s, main_only=False):
    nv, nf = s["videos"], len(s["folds"])
    L = [f"Video-level re-analysis (from saved labels, no re-decoding). Per video the metric is averaged "
         f"over the {len(s['runs'])} runs (the seeds share the folds). Paired two-sided Wilcoxon vs "
         f"{s['reference']}: over videos (n = {nv}) and over outer folds (n = {nf}, fold mean of the "
         f"run-averaged video metric; with n = 5 the smallest attainable p is 0.0625). Pooled = counts summed "
         f"over all {nv} videos of a run, then mean over runs. Max |diff| vs units/ F1@50: "
         f"{s['max_abs_diff_vs_units_F1@50']:.1e}.", ""]
    names = [v for v in s["variants"] if not main_only or v in VL_MAIN]
    for sc, title in (("all", "with Idle"), ("noidle", "without Idle")):
        L += [f"### F1@50, {title}", "",
              "| variant | mean | median delta | improved/worse/tied | p videos | mean fold delta | folds improved "
              "| p folds | pooled F1@50 (delta) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for v in names:
            m, pl = s["variants"][v][sc]["F1@50"], s["variants"][v][sc]["pooled"]["pooled_F1@50"]
            if v == s["reference"]:
                L.append(f"| {v} | {m['mean']:.1f} | - | - | - | - | - | - | {pl['mean_runs']:.1f} |")
                continue
            L.append(f"| {v} | {m['mean']:.1f} | {m['delta_median']:+.1f} | "
                     f"{m['n_improved']}/{m['n_worse']}/{m['n_tied']} | {m['wilcoxon_p_videos']:.1e} | "
                     f"{m['fold_delta_mean']:+.1f} | {m['fold_n_improved']}/{nf} | {m['wilcoxon_p_folds']:.4f} | "
                     f"{pl['mean_runs']:.1f} ({pl['delta']:+.1f}) |")
        L.append("")
        if main_only:
            break
    if main_only:
        return "\n".join(L)
    others = [k for k in VL_METRICS if k != "F1@50"]
    L += ["### Other metrics, with Idle: median per-video delta vs P0_argmax [improved/worse], Wilcoxon p (videos)",
          "", "| variant | " + " | ".join(others) + " |", "|---" * (len(others) + 1) + "|"]
    for v in names:
        if v == s["reference"]:
            continue
        cells = []
        for k in others:
            m = s["variants"][v]["all"][k]
            nd = 1 if k.startswith("F1") or k == "edit" else 3
            cells.append(f"{m['delta_median']:+.{nd}f} [{m['n_improved']}/{m['n_worse']}] p={m['wilcoxon_p_videos']:.1e}")
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    for sc, title in (("all", "with Idle"), ("noidle", "without Idle")):
        L += ["", f"### Pooled metrics (counts over all videos of a run, mean of the runs), {title}", "",
              "| variant | " + " | ".join(VL_POOLED) + " | mean over units of within-unit pooled F1@50 |",
              "|---" * (len(VL_POOLED) + 2) + "|"]
        for v in names:
            e = s["variants"][v][sc]
            cells = [f"{e['pooled'][k]['mean_runs']:.1f}" if "F1@" in k else f"{e['pooled'][k]['mean_runs']:.3f}"
                     for k in VL_POOLED]
            L.append(f"| {v} | " + " | ".join(cells) + f" | {e['pooled_within_unit_F1@50_mean']:.1f} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="dirs with phase_posteriors/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--videolevel", action="store_true",
                    help="only write summary_videolevel.{json,md} from the saved labels/ (no decoding)")
    args = ap.parse_args()
    out = Path(args.out)
    if args.videolevel:
        summarize_videolevel(args.runs, out, args.variants)
        return
    (out / "units").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        common = os.path.commonpath([str(Path(r).resolve()) for r in args.runs])
        tasks = []
        for run in args.runs:
            name = os.path.relpath(Path(run).resolve(), Path(common).parent) if len(args.runs) > 1 else Path(run).name
            folds = sorted({d["fold"] for d in load(run)[0].values()})
            for k in folds:
                for v in args.variants:
                    f = out / "units" / f"{name.replace('/', '_')}__fold{k}__{v}.json"
                    if not f.exists():
                        tasks.append((run, name, k, v, str(out)))
        # slow variants first
        cost = {"perphase": 0, "grid": 1, None: 2}
        tasks.sort(key=lambda t: (cost[VARIANTS[t[3]][2]], VARIANTS[t[3]][0] != "pelt"))
        print(f"{len(tasks)} units to run", flush=True)
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_unit, tasks):
                print(*r, flush=True)
    summarize(out, args.variants)


if __name__ == "__main__":
    main()
