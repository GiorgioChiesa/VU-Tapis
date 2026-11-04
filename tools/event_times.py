#!/usr/bin/env python3
"""
Experiment 5: the fractional time of every Orsi event used for training and evaluation.

The per-second labels (`Label/RARPxx_all_labels.csv`) put each event on the second
``round(t)`` of its raw stamp ``t`` (`data/check_webapp/main.py`, Python ``round``), and the
training table (`eval_untrimmed.build_label_table`) then drops and collapses event names.
Here every event of that table, ``(local class, second)`` exactly as `train_hybrid.load_orsi`
returns it, gets back the raw stamp it came from in
`annotations/RARPxx/SURGICAL_PHASES/RARPxx.json`:

  1. table row -> raw CSV row of the same patient and second (raw name, before collapsing);
     when several events share a second, the raw name whose collapsed name is the row's;
  2. raw CSV row -> JSON stamp(s) with the same raw name and ``round(t) == second``
     (several identical-name stamps on one second: their mean).

Rows without a stamp keep ``time = second`` and ``matched = false``. Output, one file per video:
``<out>/RARPxx.json`` = ``{"patient", "events": [{"class", "name", "second", "time", "matched"}]}``
(in the order of `load_orsi`), plus ``<out>/summary.json`` with aggregate match rates only.

Example:
    python tools/event_times.py --run outputs/orsi/LONG/cv_ft/fold0 \
        --out outputs/orsi/EXP5/event_times
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from eval_untrimmed import build_label_table, global_to_local, load_cfg

ANN_DIR = Path("/home/gchie/workspace/nas_private/data/orsi/annotations")
FRAME_DIR = "/home/gchie/workspace/data/orsi_tensors"


def raw_stamps(json_path):
    """``{raw event name: [t_sec, ...]}`` over all categories of a SURGICAL_PHASES json."""
    data = json.loads(Path(json_path).read_text())
    out = defaultdict(list)
    for evs in data["EVENTS"].values():
        for name, times in evs.items():
            out[name].extend(float(t) for t in times or [])
    return out


def stamp_for(stamps, name, second):
    """
    Fractional time of the raw stamp(s) of ``name`` that the labels put on ``second``.

    >>> stamp_for({"A": [3.4, 10.5, 11.5], "B": [3.6]}, "A", 3)
    3.4
    >>> stamp_for({"A": [3.4, 10.5, 11.5]}, "A", 12)   # Python round: 11.5 -> 12, 10.5 -> 10
    11.5
    >>> stamp_for({"A": [3.4]}, "A", 4) is None
    True
    """
    hits = [t for t in stamps.get(name, []) if int(round(t)) == int(second)]
    return float(np.mean(hits)) if hits else None


def name_map(raw, table):
    """
    Raw CSV event name -> collapsed table name, learnt on the (patient, second) pairs that hold
    exactly one event in both tables.

    >>> raw = pd.DataFrame({"patient_name": ["P", "P"], "second": [1, 2], "event_name": ["a_left", "b"]})
    >>> tab = pd.DataFrame({"patient_name": ["P", "P"], "second": [1, 2], "event_name": ["a", "b"]})
    >>> name_map(raw, tab)
    {'a_left': 'a', 'b': 'b'}
    """
    key = ["patient_name", "second"]
    r1 = raw[~raw.duplicated(key, keep=False)]
    t1 = table[~table.duplicated(key, keep=False)]
    j = r1.merge(t1, on=key, suffixes=("_raw", "_tab"))
    pairs = Counter(zip(j["event_name_raw"], j["event_name_tab"]))
    out = {}
    for (r, t), _ in pairs.most_common():
        out.setdefault(r, t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/orsi/LONG/cv_ft/fold0", help="run giving the class vocabulary")
    ap.add_argument("--ckpt", default="checkpoint_best_steps.pyth")
    ap.add_argument("--frame-dir", default=FRAME_DIR)
    ap.add_argument("--ann-dir", default=str(ANN_DIR))
    ap.add_argument("--out", default="outputs/orsi/EXP5/event_times")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(run, args.ckpt)
    patients = [
        Path(f).name.removesuffix(".csv")
        for s in ("TRAIN", "VAL", "TEST")
        for f in getattr(cfg.ENDOVIS_DATASET, f"{s}_LISTS")
    ]
    (out / "_label_table").mkdir(exist_ok=True)
    table, ids = build_label_table(cfg, patients, args.frame_dir, out / "_label_table")
    g2l, names = global_to_local(run, "steps", ids["steps"])
    events = table[(table["event_name"].str.strip().str.lower() != "idle") & table["event_id"].isin(g2l)]
    raw = pd.concat(
        [pd.read_csv(Path(args.frame_dir) / p / "Label" / f"{p}_all_labels.csv") for p in patients],
        ignore_index=True,
    )
    raw = raw[raw["event_name"].str.strip().str.lower() != "idle"]
    nmap = name_map(raw, events)

    stats = Counter()
    offsets = []
    for p in patients:
        stamps = raw_stamps(Path(args.ann_dir) / p / "SURGICAL_PHASES" / f"{p}.json")
        rp = raw[raw["patient_name"] == p]
        by_sec = {s: list(g["event_name"]) for s, g in rp.groupby("second")}
        rows = []
        for r in events[events["patient_name"] == p].itertuples():
            cands = by_sec.get(r.second, [])
            if len(cands) > 1:
                cands = [c for c in cands if nmap.get(c) == r.event_name] or cands
                stats["shared_second"] += 1
            t = None
            for c in cands:
                t = stamp_for(stamps, c, r.second)
                if t is not None:
                    break
            stats["events"] += 1
            stats["matched"] += t is not None
            if t is not None:
                offsets.append(t - r.second)
            rows.append(
                {
                    "class": int(g2l[r.event_id]),
                    "name": names[g2l[r.event_id]],
                    "second": int(r.second),
                    "time": float(t if t is not None else r.second),
                    "matched": t is not None,
                }
            )
        (out / f"{p}.json").write_text(json.dumps({"patient": p, "events": rows}, indent=1))
    offsets = np.asarray(offsets)
    summary = {
        "run": str(run),
        "patients": len(patients),
        "events": stats["events"],
        "matched": stats["matched"],
        "match_rate": stats["matched"] / max(stats["events"], 1),
        "shared_second_rows": stats["shared_second"],
        "offset_mean": float(offsets.mean()),
        "offset_abs_max": float(np.abs(offsets).max()),
        "integer_share": float(np.mean(offsets == 0)),
        "offset_hist_0.1": np.histogram(offsets, bins=np.linspace(-0.5, 0.5, 11))[0].tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
