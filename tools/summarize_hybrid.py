#!/usr/bin/env python3
"""
Experiment 1 table: global-context configurations of `tools/train_hybrid.py`, and the winner H*.

Runs are ``<root>/<config>/seed<s>/fold<i>/val`` with `report_nms5.json` (eval_untrimmed),
`hybrid_metrics.json` (train_hybrid) and, for G1, `constraints_nms5.json`. Config names are
``<G><c?>_<target>`` (``c`` = causal). G7 is not trained: it is G1 decoded with the caps +
order chain priors of eval_constraints (``caps_chain|a=0.1|s=1|f=0.2``), G7p the same plus
gating by the predicted phases. When other configs also have `constraints_nms5.json`, their
caps + chain decoding is reported as ``<G><c?>+cc_<target>`` (information only, not a
candidate of the decision rule). Every metric is the mean +/- std over the (seed, fold) runs.

Decision rule (fixed before running, plan "Esperimento 1"), on the dilate3 configs:
  1. feasible: mean phase accuracy >= that of G2 - 0.01;
  2. leader: highest mean mAP@5 among the feasible configs;
  3. tied with the leader: mean mAP@5 within 1 std (of the leader's runs) of it;
  4. among the tied configs a causal one is preferred, then the simplest
     (COMPLEXITY order), then the highest avg-mAP tight. That config is H*.
Paired Wilcoxon signed-rank tests (same seed and fold) are reported against G2 and G2c.

Input ablation runs (``--streams static|two``) are named ``<config>-<streams>`` and tested
against the same config on dynamic features; they are not candidates of the rule.

``--dataset grasp`` summarises the second view (``outputs/grasp/HYBRID/<config>/seed<s>/val``,
GraSP train -> test): metrics of `hybrid_metrics.json` (same spotting definitions,
computed by train_hybrid), mean +/- std over seeds; the rule is applied for information.

Example:
    python tools/summarize_hybrid.py --root outputs/orsi/HYBRID [--wandb]
    python tools/summarize_hybrid.py --dataset grasp
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

G7_KEYS = {"G7": "caps_chain|a=0.1|s=1|f=0.2", "G7p": "phase_caps_chain|a=0.1|s=1|f=0.2"}
# simplest first; G7 = G1 + decoding rules estimated by counting (no extra learning)
COMPLEXITY = ["G0", "G1", "G7", "G7p", "G3", "G2", "G4", "G5", "G6"]
COLS = [
    ("mAP@1", "mAP@1"), ("mAP@2", "mAP@2"), ("mAP@5", "mAP@5"), ("mAP@10", "mAP@10"), ("mAP@30", "mAP@30"),
    ("tight", "tight"), ("loose", "loose"), ("F1@5", "F1@5"), ("FA/h", "FA/h"), ("phase_acc", "ph.acc"),
    ("phase_edit", "edit"), ("phase_F1@10", "sF1@10"), ("phase_F1@25", "sF1@25"), ("phase_F1@50", "sF1@50"),
]
GRASP_COLS = [
    ("mAP@1", "mAP@1"), ("mAP@5", "mAP@5"), ("mAP@30", "mAP@30"), ("tight", "tight"), ("loose", "loose"),
    ("F1@5", "F1@5"), ("FA/h", "FA/h"), ("onset_step_mAP@5", "step on. mAP@5"),
    ("onset_phase_mAP@5", "phase on. mAP@5"), ("phase_acc", "ph.acc"), ("phase_edit", "edit"),
    ("phase_F1@50", "sF1@50"),
]
NAME = re.compile(r"^(G\dp?)(c?)(\+cc)?_([a-z]+\d*)(-\w+)?$")  # context, causal, +cc, target, -streams


def run_metrics(val):
    rep = json.loads((val / "report_nms5.json").read_text())
    hyb = json.loads((val / "hybrid_metrics.json").read_text())
    nms = rep["spotting_nms"]
    op = nms["operating_points"]["delta_5"]["best_f1"]
    m = {f"mAP@{d}": nms["map"][str(d)] for d in (1, 2, 5, 10, 30)}
    m.update(
        tight=nms["avg_map_tight_1_5s"], loose=nms["avg_map_loose_5_30s"], **{"F1@5": op["f1"]},
        **{"FA/h": op["false_alarms_per_hour"]}, phase_acc=rep["phases_untrimmed"]["accuracy"],
        params=hyb["n_params"], train_min=hyb["train_time_s"] / 60, latency_ms_h=hyb["latency_ms_per_video_hour"],
        **{k: hyb[k] / 100 for k in ("phase_edit", "phase_F1@10", "phase_F1@25", "phase_F1@50")},
    )
    derived = {}
    cpath = val / "constraints_nms5.json"
    if cpath.exists():
        res = json.loads(cpath.read_text())["results"]
        for g, key in G7_KEYS.items():
            r = res[key]
            derived[g] = dict(
                m, **{f"mAP@{d}": r["map"][str(d)] for d in (1, 2, 5, 10, 30)}, tight=r["avg_map_tight"],
                loose=r["avg_map_loose"], **{"F1@5": r["best_f1_5s"]["f1"]},
                **{"FA/h": r["best_f1_5s"]["false_alarms_per_hour"]},
            )
    return m, derived


def hybrid_only_metrics(val):
    """Metrics from `hybrid_metrics.json` alone (datasets without eval_untrimmed)."""
    hyb = json.loads((val / "hybrid_metrics.json").read_text())
    m = {f"mAP@{d}": hyb[f"mAP@{d}s"] for d in (1, 2, 5, 10, 30)}
    m.update(
        tight=hyb["avg_mAP_tight"], loose=hyb["avg_mAP_loose"], **{"F1@5": hyb["F1@5s"]},
        **{"FA/h": hyb["FA_per_hour@5s"]}, phase_acc=hyb["phase_accuracy"], params=hyb["n_params"],
        train_min=hyb["train_time_s"] / 60, latency_ms_h=hyb["latency_ms_per_video_hour"],
        **{k: hyb[k] / 100 for k in ("phase_edit", "phase_F1@10", "phase_F1@25", "phase_F1@50")},
        **{k.removesuffix("s"): v for k, v in hyb.items() if k.startswith("onset_") and k.endswith("s")},
    )
    return m


def collect_seeds(root):
    """``{config: {(seed, "-"): metrics}}`` for train -> test runs without folds."""
    runs = defaultdict(dict)
    for val in sorted(root.glob("*/seed*/val")):
        if (val / "hybrid_metrics.json").exists():
            cfg, seed = val.parts[-3:-1]
            runs[cfg][(seed, "-")] = hybrid_only_metrics(val)
    return runs


def collect(root):
    """``{config: {(seed, fold): metrics}}``"""
    runs = defaultdict(dict)
    for val in sorted(root.glob("*/seed*/fold*/val")):
        if not (val / "report_nms5.json").exists() or not (val / "hybrid_metrics.json").exists():
            continue
        cfg, seed, fold = val.parts[-4:-1]
        m, derived = run_metrics(val)
        runs[cfg][(seed, fold)] = m
        g, c, _, tgt, st = NAME.match(cfg).groups()
        st = st or ""
        for d, dm in derived.items():
            if g == "G1":
                runs[f"{d}{c}_{tgt}{st}"][(seed, fold)] = dm
            elif d == "G7":
                runs[f"{g}{c}+cc_{tgt}{st}"][(seed, fold)] = dm
    return runs


def paired_p(a, b, key):
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None, len(common)
    x = np.array([a[k][key] for k in common])
    y = np.array([b[k][key] for k in common])
    if np.allclose(x, y):
        return 1.0, len(common)
    return float(wilcoxon(x, y).pvalue), len(common)


def table(runs):
    out = {}
    for cfg, rs in runs.items():
        keys = set.intersection(*(set(m) for m in rs.values()))
        out[cfg] = {
            k: (float(np.mean([m[k] for m in rs.values()])), float(np.std([m[k] for m in rs.values()], ddof=1)) if len(rs) > 1 else 0.0)
            for k in keys
        }
        out[cfg]["n_runs"] = (len(rs), 0.0)
    return out


def decide(tab, runs, target="dilate3", causal_only=False, min_runs=5):
    """H* by the decision rule; ``causal_only`` applies it to the causal configs with G2c as reference."""
    cands = {
        c: r for c, r in tab.items()
        if c.endswith(f"_{target}") and "+cc" not in c and r["n_runs"][0] >= min_runs
        and (not causal_only or NAME.match(c).group(2) == "c")
    }
    ref_name = f"G2{'c' if causal_only else ''}_{target}"
    ref = cands.get(ref_name)
    if ref is None:
        return None, [f"no {ref_name} reference"]
    notes = []
    feasible = {c: r for c, r in cands.items() if r["phase_acc"][0] >= ref["phase_acc"][0] - 0.01}
    for c in sorted(set(cands) - set(feasible)):
        notes.append(
            f"{c} excluded: phase acc {cands[c]['phase_acc'][0]:.3f} < {ref_name} {ref['phase_acc'][0]:.3f} - 0.01"
        )
    leader = max(feasible, key=lambda c: feasible[c]["mAP@5"][0])
    lm, ls = feasible[leader]["mAP@5"]
    tied = [c for c, r in feasible.items() if lm - r["mAP@5"][0] < ls]
    notes.append(f"leader {leader}: mAP@5 {lm:.3f} +/- {ls:.3f}; tied within 1 std: {sorted(tied)}")
    causal = [c for c in tied if NAME.match(c).group(2) == "c"]
    pool = causal or tied
    if causal:
        notes.append(f"causal configs among the tied: {sorted(causal)}")
    rank = lambda c: COMPLEXITY.index(NAME.match(c).group(1))  # noqa: E731
    best_rank = min(rank(c) for c in pool)
    pool = [c for c in pool if rank(c) == best_rank]
    winner = max(pool, key=lambda c: feasible[c]["tight"][0])
    notes.append(f"{'causal ' if causal_only else ''}H* = {winner}")
    return winner, notes


def fmt(v, s, digits=3):
    return f"{v:.{digits}f} ± {s:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="orsi", choices=["orsi", "grasp"])
    parser.add_argument("--root", help="default: outputs/<dataset>/HYBRID")
    parser.add_argument("--out", help="default: <root>/summary.{json,md}")
    parser.add_argument("--wandb", action="store_true", help="write the eval_untrimmed numbers to the W&B run summaries")
    parser.add_argument("--wandb-path", default="eidos-giorgio/OrsiTapis")
    args = parser.parse_args()

    grasp = args.dataset == "grasp"
    root = Path(args.root or f"outputs/{args.dataset}/HYBRID")
    runs = collect_seeds(root) if grasp else collect(root)
    cols = GRASP_COLS if grasp else COLS
    tab = table(runs)
    pvals = {}
    for cfg in tab:
        tgt = NAME.match(cfg).group(4)
        for ref in (f"G2_{tgt}", f"G2c_{tgt}"):
            if ref in runs and ref != cfg:
                for key in ("mAP@5", "tight", "phase_acc"):
                    pvals[(cfg, ref, key)] = paired_p(runs[cfg], runs[ref], key)

    def order(c):
        g, causal, cc, tgt, st = NAME.match(c).groups()
        return tgt, st or "", causal, COMPLEXITY.index(g), bool(cc), c

    cfgs = sorted(tab, key=order)
    lines = [
        "| config | runs | params (M) | " + " | ".join(h for _, h in cols) + " | p mAP@5 vs G2 | p vs G2c | train (min) | latency (ms / video h) |",
        "|---" * (len(cols) + 7) + "|",
    ]
    for cfg in cfgs:
        r = tab[cfg]
        tgt = NAME.match(cfg).group(4)
        p1 = pvals.get((cfg, f"G2_{tgt}", "mAP@5"), (None, 0))[0]
        p2 = pvals.get((cfg, f"G2c_{tgt}", "mAP@5"), (None, 0))[0]
        lines.append(
            f"| {cfg} | {r['n_runs'][0]} | {r['params'][0] / 1e6:.3f} | "
            + " | ".join(fmt(*r[k], 1 if k == "FA/h" else 3) for k, _ in cols)
            + f" | {'—' if p1 is None else f'{p1:.3f}'} | {'—' if p2 is None else f'{p2:.3f}'}"
            + f" | {r['train_min'][0]:.1f} | {r['latency_ms_h'][0]:.1f} |"
        )
    md = "\n".join(lines)
    min_runs = 3 if grasp else 5
    winner, notes = decide(tab, runs, min_runs=min_runs)
    md += "\n\nDecision (dilate3):\n" + "\n".join(f"- {n}" for n in notes)
    winner_c, notes_c = decide(tab, runs, causal_only=True, min_runs=min_runs)
    md += "\n\nCausal track (dilate3, reference G2c):\n" + "\n".join(f"- {n}" for n in notes_c)
    print(md)

    out = Path(args.out) if args.out else root / "summary"
    out.with_suffix(".md").write_text(md + "\n")
    out.with_suffix(".json").write_text(json.dumps({
        "table": {c: {k: list(v) for k, v in r.items()} for c, r in tab.items()},
        "wilcoxon": {f"{c}|{ref}|{k}": {"p": p, "n": n} for (c, ref, k), (p, n) in pvals.items()},
        "decision": {"winner": winner, "notes": notes, "causal_winner": winner_c, "causal_notes": notes_c},
    }, indent=2))
    print(f"table: {out}.md / .json")

    if args.wandb:
        import wandb

        api = wandb.Api()
        by_name = {r.name: r for r in api.runs(args.wandb_path, filters={"jobType": "hybrid"})}
        n = 0
        prefix = "hybrid-grasp" if grasp else "hybrid"
        for cfg, rs in runs.items():
            for (seed, fold), m in rs.items():
                run = by_name.get(f"{prefix}/{cfg}/{seed}" + ("" if fold == "-" else f"/{fold}"))
                if run is not None:
                    run.summary.update({f"full_eval/{k}": v for k, v in m.items()})
                    run.update()
                    n += 1
        print(f"updated {n} W&B runs")


if __name__ == "__main__":
    main()
