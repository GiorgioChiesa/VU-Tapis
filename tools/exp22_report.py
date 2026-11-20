#!/usr/bin/env python3
"""
EXP-22 evaluation of one fold: L-A3 on the EXP-19 arm S latents (tools/train_long_s.py) against the reference S MS-TCN
(``reference_s{0,1}`` from tools/s_fold_inputs.py), with the EXP-21 evaluator (tools/p3lib.py -> tools/eval_focus8.py).

Per model and decoder (nms, caps_chain), on the 10 held-out videos: AP per class and group mAP at +-1/2/5 s with a
per-video bootstrap CI (1000), phase accuracy / F1@50 for raw argmax (P0) and hysteresis (P1), per-video paired
Wilcoxon tests of L-A3 (mean of 2 seeds per video) against the reference (mean of 2 seeds per video). Fold 0 also
evaluates, with the same code, EXP-21 arm M models (L-A3 long head, short MS-TCN) and, with ``--online``, the causal
L-A3 (online decoding as tools/online_eval_focus8.py: clip delay 2 s, decoder delay 2 s; the causal model, seed or
seed ensemble, and decoder are selected on the inner val). Aggregates only; nothing per patient is written.

    python tools/exp22_report.py --fold 0 --online
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F, ev  # noqa: E402

EXP22 = P.REPO / "results/exp22"
FOCUS8P3 = Path("/home/gchie/workspace/wt/focus8-p3")
M_LONG = P.LONG_DIR
M_SHORT = FOCUS8P3 / "results/p3/short_mstcn"
NAMES9 = F.EVENT_CLASSES
DECODERS = ("nms", "caps_chain")
GROUPS = ("short", "long")


def load(run_dir, ctx):
    d = Path(run_dir) / ctx.set
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    names = meta.get("event_classes") or NAMES9
    return P.load_dir(d, ctx.videos), names


def ensemble(runs):
    """Mean of the probabilities of several loaded runs (same rows)."""
    out = {}
    for v in runs[0]:
        assert all(np.array_equal(r[v]["second"], runs[0][v]["second"]) for r in runs)
        out[v] = {"second": runs[0][v]["second"], **{k: np.mean([r[v][k] for r in runs], 0)
                                                    for k in ("probs_steps", "probs_phases")}}
    return out


def seed_mean(pv_list):
    """Per-video mean over seeds of ``{video: value}`` dicts."""
    return {v: float(np.nanmean([d[v] for d in pv_list])) for v in pv_list[0]}


def evaluate(runs, ctx, n_boot):
    """runs: [(preds, names)] of the seeds. Returns {decoder: [per-seed results]} and the phases per seed."""
    res = {d: [] for d in DECODERS}
    phs = []
    gt_cache = {}
    for preds, names in runs:
        key = tuple(names)
        gt_cache.setdefault(key, ctx.gts(names))
        for d in DECODERS:
            dets = ctx.detect(preds, names, d)
            res[d].append(P.eval_events(dets, gt_cache[key], names, ctx.videos, n_boot, 0))
        phs.append(P.eval_phases(preds, ctx)[:2] if "probs_phases" in preds[ctx.videos[0]] else None)
    return res, phs


def group_value(r, g, d):
    return r["group_map"][g][str(d)] if g in r["group_map"] else float("nan")


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def f3(x, ci=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = EXP22 / f"fold{a.fold}"
    out = Path(a.out) if a.out else R / "report"
    P.SPLITS = R / "splits.json"
    ct = P.Ctx("held_out")
    models = {  # name -> [(run dir)] per seed
        "S L-A3 offline": [R / f"a3_s{s}" for s in (0, 1)],
        "S MS-TCN reference": [R / f"reference_s{s}" for s in (0, 1)],
    }
    if a.fold == 0:
        models["M L-A3 offline (EXP-21, long head)"] = [M_LONG / f"a3_s{s}" for s in (0, 1)]
        models["M short MS-TCN (EXP-21 p3)"] = [M_SHORT / f"s{s}" for s in (0, 1)]
    res, phs, per_seed_dirs = {}, {}, {}
    for name, dirs in models.items():
        runs = []
        for d in dirs:
            if not (Path(d) / "held_out").exists() and "reference" not in name:
                continue
            runs.append(load(d, ct) if "reference" not in name else
                        (P.load_dir(d, ct.videos), json.loads((d / "meta.json").read_text())["event_classes"]))
        res[name], phs[name] = evaluate(runs, ct, a.boot)
        print("evaluated", name, len(runs), "seeds", flush=True)
    pdict = {n: [d.name for d in ds] for n, ds in models.items()}

    # ---- tables
    md = [f"# EXP-22 fold {a.fold}: L-A3 on EXP-19 S latents vs reference (held-out, 10 videos, 95% CI = per-video bootstrap)\n"]
    for dec in DECODERS:
        rows = []
        for name in res:
            for s, r in enumerate(res[name][dec]):
                ci = r.get("group_map_ci95", {})
                cells = [" / ".join(f3(group_value(r, g, d), ci.get(g, {}).get(str(d))) for d in P.DELTAS) for g in GROUPS]
                rows.append([name, f"seed {s}", *cells])
            if len(res[name][dec]) > 1:
                mean = [" / ".join(f3(float(np.nanmean([group_value(r, g, d) for r in res[name][dec]]))) for d in P.DELTAS)
                        for g in GROUPS]
                rows.append([name, "mean of seeds", *mean])
        md += [f"## Group mAP, decoder {dec} (mAP@1 / @2 / @5)", md_table(["model", "run", "short", "long"], rows), ""]
    for dec in DECODERS:
        rows = []
        for name in res:
            for s, r in enumerate(res[name][dec]):
                ci = r.get("per_class_ap_ci95", {})
                for cl in NAMES9[1:]:
                    if cl in r["per_class_ap"]["2"]:
                        rows.append([name, f"seed {s}", cl, str(r["n_gt"][cl]), " / ".join(
                            f3(r["per_class_ap"][str(d)][cl], ci.get(str(d), {}).get(cl)) for d in P.DELTAS)])
        md += [f"## AP per class, decoder {dec} (AP@1 / @2 / @5)", md_table(["model", "run", "class", "n_gt", "AP"], rows), ""]
    rows = []
    for name in phs:
        for s, p in enumerate(phs[name]):
            if p is None:
                continue
            agg = p[0]
            rows.append([name, f"seed {s}", f3(agg["row_accuracy"]), f3(agg["row_macro_f1"]),
                         f"{agg['F1@50_mean_video']['P0']:.1f}", f"{agg['F1@50_mean_video']['P1']:.1f}",
                         f3(agg["acc_1hz_mean_video"]["P0"]), f3(agg["acc_1hz_mean_video"]["P1"])])
    md += ["## Phases (held-out)", md_table(["model", "run", "row acc", "row macro F1", "F1@50 P0", "F1@50 P1",
                                              "acc 1 Hz P0", "acc 1 Hz P1"], rows), ""]

    # ---- paired Wilcoxon (per video, seed means) and the prediction check
    test = {}
    S, REF = "S L-A3 offline", "S MS-TCN reference"
    others = [REF] + [n for n in res if n not in (S, REF)]
    for dec in DECODERS:
        for other in others:
            for g in GROUPS:
                if g not in res[other][dec][0]["per_video_group_ap"]:
                    continue
                for d in P.DELTAS:
                    x = seed_mean([r["per_video_group_ap"][g][str(d)] for r in res[S][dec]])
                    y = seed_mean([r["per_video_group_ap"][g][str(d)] for r in res[other][dec]])
                    test[f"{dec}|{S} vs {other}|{g}|{d}"] = P.wilcoxon(x, y)
    for other in [REF]:
        for k in ("P0", "P1"):
            x = seed_mean([p[1]["F1@50"][k] for p in phs[S]])
            y = seed_mean([p[1]["F1@50"][k] for p in phs[other]])
            test[f"phases F1@50 {k}|{S} vs {other}"] = P.wilcoxon(x, y)
    md += ["## Paired Wilcoxon over the 10 held-out videos (L-A3 S, mean of seeds per video, minus the other model)",
           md_table(["key", "n", "mean diff", "median diff", "better", "worse", "p"],
                    [[k, str(t["n_videos"]), f"{t['mean_diff']:+.3f}", f"{t['median_diff']:+.3f}", str(t["n_better"]),
                      str(t["n_worse"]), f"{t['p']:.3f}"] for k, t in test.items()]), ""]
    pred = {}
    for dec in DECODERS:
        s_vals = [group_value(r, "long", 2) for r in res[S][dec]]
        r_vals = [group_value(r, "long", 2) for r in res[REF][dec]]
        pred[dec] = {"S_L-A3_long_map2_seeds": s_vals, "S_L-A3_mean": float(np.mean(s_vals)),
                     "reference_seeds": r_vals, "reference_mean": float(np.mean(r_vals)),
                     "threshold_ref_plus_0.03": float(np.mean(r_vals)) + 0.03,
                     "difference": float(np.mean(s_vals) - np.mean(r_vals))}
    md += ["## Prediction check: long-group mAP@+-2 s, mean of 2 seeds, S L-A3 offline vs reference + 0.03",
           md_table(["decoder", "S L-A3 seeds", "S L-A3 mean", "reference seeds", "reference mean", "threshold", "difference"],
                    [[d, ", ".join(f3(x) for x in v["S_L-A3_long_map2_seeds"]), f3(v["S_L-A3_mean"]),
                      ", ".join(f3(x) for x in v["reference_seeds"]), f3(v["reference_mean"]),
                      f3(v["threshold_ref_plus_0.03"]), f"{v['difference']:+.3f}"] for d, v in pred.items()]), ""]
    report = {"fold": a.fold, "results": res, "phases": {n: [None if p is None else p[0] for p in ps] for n, ps in phs.items()},
              "phases_per_video": {n: [None if p is None else p[1] for p in ps] for n, ps in phs.items()},
              "wilcoxon": test, "prediction_check": pred, "runs": pdict, "splits": str(P.SPLITS)}

    # ---- online (causal L-A3)
    if a.online:
        import online_eval_focus8 as O

        cv = P.Ctx("inner_val")
        cands = {}
        for st, c in (("inner_val", cv), ("held_out", ct)):
            rs = [load(R / f"a3_causal_s{s}", c)[0] for s in (0, 1)]
            cands[st] = {"a3_causal_s0": rs[0], "a3_causal_s1": rs[1], "a3_causal_ens": ensemble(rs)}
        gts = {c.set: c.gts(NAMES9) for c in (cv, ct)}
        classes = list(range(1, len(NAMES9)))

        def emissions(c, preds, var):
            pri = c.priors(NAMES9)
            return {v: O.drop_ign(O.decode(preds[v]["probs_steps"], preds[v]["second"], classes, var, pri, 2.0), c.ign[v])
                    for v in c.videos}

        def spot(c, em, nb=0):
            dets = {v: [(k, t, s) for k, t, s, _ in e] for v, e in em.items()}
            return P.eval_events(dets, gts[c.set], NAMES9, c.videos, nb, 0)

        best = None
        sel = {}
        for cand in cands["inner_val"]:
            for var in O.VARIANTS:
                r = spot(cv, emissions(cv, cands["inner_val"][cand], var))
                sc = float(np.mean([P.score(r, "short"), P.score(r, "long")]))
                sel[f"{cand}|{var}"] = sc
                if best is None or sc > best[0] + 1e-12:
                    best = (sc, cand, var)
        em_t = emissions(ct, cands["held_out"][best[1]], best[2])
        ron = spot(ct, em_t, a.boot)
        lat = {str(d): O.latency(em_t, gts["held_out"], d, 2.0) for d in P.DELTAS}
        # phases: seed / ensemble with the best inner-val hysteresis F1@50
        bp = max(cands["inner_val"], key=lambda k: O.phases_online(cands["inner_val"][k], cv)[0]["F1@50_mean_video"]["H"])
        pon = O.phases_online(cands["held_out"][bp], ct)[0]
        # the per-seed online results too (no selection), nms-like plain decoder
        per_seed = {}
        for cand in ("a3_causal_s0", "a3_causal_s1"):
            per_seed[cand] = {var: spot(ct, emissions(ct, cands["held_out"][cand], var)) for var in O.VARIANTS}
        report["online"] = {"selection_inner_val_score": sel, "selected": {"cand": best[1], "decoder": best[2], "inner_score": best[0]},
                            "test_events": ron, "latency": lat, "phases_cand": bp, "phases": pon,
                            "per_seed_test": {k: {var: {"group_map": x["group_map"]} for var, x in v.items()} for k, v in per_seed.items()}}
        gm, ci = ron["group_map"], ron.get("group_map_ci95", {})
        md += [f"## Online (causal L-A3 S; selected on inner val: {best[1]}, {best[2]}; clip delay 2 s, decoder delay 2 s)",
               md_table(["group", "mAP@1 / @2 / @5 [95% CI]"],
                        [[g, " / ".join(f3(gm[g][str(d)], ci.get(g, {}).get(str(d))) for d in P.DELTAS)] for g in gm]),
               "", "Latency of the true positives (s): " + "; ".join(
                   f"+-{d}: mean {l['mean']:.2f} std {l['std']:.2f} n {l['n_tp']}" for d, l in lat.items()),
               "", f"Phases online ({bp}): F1@50 P0 {pon['F1@50_mean_video']['P0']:.1f}, H {pon['F1@50_mean_video']['H']:.1f}; "
               f"acc 1 Hz P0 {pon['acc_1hz_mean_video']['P0']:.3f}, H {pon['acc_1hz_mean_video']['H']:.3f}", ""]
        rows = [[k, var] + [" / ".join(f3(x["group_map"][g][str(d)]) for d in P.DELTAS) for g in GROUPS]
                for k, v in per_seed.items() for var, x in v.items()]
        md += ["Per seed (no selection), online group mAP@1 / @2 / @5:", md_table(["run", "decoder", "short", "long"], rows), ""]
        e21 = FOCUS8P3 / "results/p3/online/online.json"
        if e21.exists():
            o = json.loads(e21.read_text())
            g21 = o["test"]["events"]["group_map"]
            report["online"]["exp21_online_test_group_map"] = g21
            md += ["EXP-21 online (arm M encoder short + causal M long model), same evaluation: " +
                   "; ".join(f"{g} " + " / ".join(f3(g21[g][str(d)]) for d in P.DELTAS) for g in g21), ""]
    out.mkdir(parents=True, exist_ok=True)
    P.dump_json(report, out / "report.json")
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
