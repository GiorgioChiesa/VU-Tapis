#!/usr/bin/env python3
"""
Collect eval_untrimmed / eval_constraints reports of the temporal runs into one table.

Runs are the sub-directories of --root containing `val/report_nms5.json`.
Cross-validation groups (`cv_*/fold*/`) are averaged over folds (mean +/- std).
With --wandb the full-evaluation numbers (including the constrained decoders) are
written to the summary of the matching W&B run (`temporal/<name>`): aggregate
scalars only.

Example:
    python tools/summarize_temporal.py --root outputs/orsi/TEMPORAL --wandb
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONSTRAINT_KEYS = {
    "caps_chain": "caps_chain|a=0.1|s=1|f=0.2",
    "phase_caps_chain": "phase_caps_chain|a=0.1|s=1|f=0.2",
    "oracle_caps_chain": "oracle_caps_chain|a=0.1|s=1|f=0.2",
}


def metrics(run_dir):
    rep = json.loads((run_dir / "val" / "report_nms5.json").read_text())
    nms = rep["spotting_nms"]
    op = nms["operating_points"]["delta_5"]["best_f1"]
    out = {
        "mAP@1": nms["map"]["1"],
        "mAP@2": nms["map"]["2"],
        "mAP@5": nms["map"]["5"],
        "mAP@10": nms["map"]["10"],
        "mAP@30": nms["map"]["30"],
        "tight": nms["avg_map_tight_1_5s"],
        "loose": nms["avg_map_loose_5_30s"],
        "F1@5": op["f1"],
        "FA/h": op["false_alarms_per_hour"],
        "phase_acc": rep["phases_untrimmed"]["accuracy"],
        "phase_mAP": rep["phases_untrimmed"]["frame_map"],
    }
    cpath = run_dir / "val" / "constraints_nms5.json"
    if cpath.exists():
        res = json.loads(cpath.read_text())["results"]
        for short, key in CONSTRAINT_KEYS.items():
            if key in res:
                out[f"{short}:mAP@5"] = res[key]["map"]["5"]
                out[f"{short}:loose"] = res[key]["avg_map_loose"]
                out[f"{short}:F1@5"] = res[key]["best_f1_5s"]["f1"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs/orsi/TEMPORAL")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-path", default="eidos-giorgio/OrsiTapis")
    parser.add_argument("--out", help="json file for the table (default: <root>/summary.json)")
    args = parser.parse_args()

    root = Path(args.root)
    runs = {
        str(p.parent.parent.relative_to(root)): metrics(p.parent.parent)
        for p in sorted(root.glob("**/val/report_nms5.json"))
    }

    groups = defaultdict(list)
    for name, m in runs.items():
        if "/fold" in name:
            groups[name.split("/fold")[0]].append(m)
    table = {n: {k: (v, None) for k, v in m.items()} for n, m in runs.items() if "/fold" not in n}
    for g, ms in groups.items():
        keys = set.intersection(*(set(m) for m in ms))
        table[f"{g} ({len(ms)} folds)"] = {
            k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in keys
        }

    cols = ["mAP@1", "mAP@5", "mAP@30", "tight", "loose", "F1@5", "FA/h", "phase_acc",
            "caps_chain:mAP@5", "caps_chain:F1@5", "oracle_caps_chain:mAP@5"]
    print(f"{'run':34s} " + " ".join(f"{c.replace('caps_chain', 'cc'):>13s}" for c in cols))
    for name in sorted(table):
        row = table[name]
        cells = []
        for c in cols:
            v, s = row.get(c, (float("nan"), None))
            cells.append(f"{v:6.3f}±{s:5.3f}" if s is not None else f"{v:13.3f}")
        print(f"{name:34s} " + " ".join(cells))

    out = Path(args.out) if args.out else root / "summary.json"
    out.write_text(json.dumps({n: {k: list(v) for k, v in r.items()} for n, r in table.items()}, indent=2))
    print(f"table: {out}")

    if args.wandb:
        import wandb

        api = wandb.Api()
        by_name = {r.name: r for r in api.runs(args.wandb_path, filters={"jobType": "temporal_head"})}
        for name, m in runs.items():
            run = by_name.get(f"temporal/{name}")
            if run is None:
                print(f"no W&B run for {name}")
                continue
            run.summary.update({f"full_eval/{k}": v for k, v in m.items()})
            run.update()
        print(f"updated {sum(f'temporal/{n}' in by_name for n in runs)} W&B runs")


if __name__ == "__main__":
    main()
