#!/usr/bin/env python3
"""
Re-read of the "gating with ground-truth phases" result (stato_eventi_TAPIS §8, +0.07 mAP@5):
is the gain on the events stamped on a phase change, like the oracle-phase input of Experiment 2?

For every run of the given arms (``<root>/<arm>/seed*/fold*/val/constraints_nms5.json``, written by
eval_constraints.py) the per-class AP@5 of each decoding is compared with plain NMS. Classes are
split into boundary events (fraction of onsets within 2 s of a phase change >= 0.5, from
``oracle_phase_boundary.json`` of tools/analyze_oracle_phase.py) and the others. Paired Wilcoxon
over the runs of the group means. Aggregate output only.

Example:
    python tools/analyze_gating_boundary.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "outputs/orsi"
SUFFIX = "|a=0.1|s=1|f=0.2"
DECODINGS = {"phase_oracle": "nms", "phase": "nms", "oracle_caps_chain": "caps_chain", "phase_caps_chain": "caps_chain"}


def group_delta(arm_ap, ref_ap, classes):
    """Mean over ``classes`` of AP(arm) - AP(ref), skipping classes without AP (None/NaN) in either.

    >>> group_delta({"a": 0.5, "b": 0.9, "c": None, "d": float("nan")}, {"a": 0.25, "b": 0.8, "c": 0.1, "d": 0.2},
    ...             ["a", "b", "c", "d"])
    0.175
    """
    ok = lambda v: v is not None and np.isfinite(v)
    d = [arm_ap[c] - ref_ap[c] for c in classes if ok(arm_ap.get(c)) and ok(ref_ap.get(c))]
    return round(float(np.mean(d)), 6) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["EXP2/F_base", "HYBRID/G2_dilate3"])
    ap.add_argument("--boundary", default=str(ROOT / "EXP2/analysis/oracle_phase_boundary.json"))
    ap.add_argument("--out", default=str(ROOT / "EXP2/analysis/gating_boundary.json"))
    args = ap.parse_args()
    rows = json.loads(Path(args.boundary).read_text())["per_class"]
    bnd = [r["event"] for r in rows if r["boundary_frac"] >= 0.5]
    oth = [r["event"] for r in rows if r["boundary_frac"] < 0.5]
    res = {"boundary_classes": bnd, "n_other": len(oth), "arms": {}}
    for arm in args.arms:
        runs = sorted((ROOT / arm).glob("seed*/fold*/val/constraints_nms5.json"))
        per = {k: {"boundary": [], "other": [], "all": []} for k in DECODINGS}
        for f in runs:
            r = json.loads(f.read_text())["results"]
            for dec, ref in DECODINGS.items():
                a, b = r[dec + SUFFIX]["per_class_ap_5s"], r[ref + SUFFIX]["per_class_ap_5s"]
                per[dec]["boundary"].append(group_delta(a, b, bnd))
                per[dec]["other"].append(group_delta(a, b, oth))
                per[dec]["all"].append(r[dec + SUFFIX]["map"]["5"] - r[ref + SUFFIX]["map"]["5"])
        out = {"n_runs": len(runs)}
        for dec, groups in per.items():
            out[f"{dec} vs {DECODINGS[dec]}"] = {
                g: {"delta": float(np.mean(v)), "p": float(wilcoxon(v).pvalue) if np.any(v) else 1.0,
                    "wins": int((np.array(v) > 0).sum())}
                for g, v in groups.items()}
        res["arms"][arm] = out
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["arms"], indent=1))


if __name__ == "__main__":
    main()
