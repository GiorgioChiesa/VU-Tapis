#!/usr/bin/env python3
"""
EXP-22: inputs of the EXP-19 arm S folds for the long-context models.

For fold ``i`` writes ``<out>/fold<i>/splits.json`` (train, inner_val, held_out) and the reference MS-TCN
predictions (EXP-19 S dilate3, seeds 0 and 1) restricted to the 9 focus8 event columns, as
``<out>/fold<i>/reference_{s0,s1}/RARPxx.npz`` + ``meta.json`` (the format of tools/eval_focus8.py):

* train / inner_val = ``train_data.csv`` / ``val_data.csv`` of the S fine-tune of the fold (made by
  cv_finetune.choose_inner_val), held_out = the videos of ``TEMPORAL/cv_ft_dilate3/fold<i>/val``;
* event columns: row order of ``distributions/train_steps_distribution.csv`` of the fold (33 classes);
* fold 0: the splits must equal the EXP-21 splits and the 9-column files must equal the existing
  ``results/reference/v2026-11-19/dilate3{,_s1}_focus8`` files (asserted).

    python tools/s_fold_inputs.py --fold 0 --out results/exp22
"""

import argparse
import json
from pathlib import Path

import numpy as np

S = Path("/home/gchie/workspace/wt/short-long-context/results/S")
EXP21_SPLITS = Path("/home/gchie/workspace/wt/focus8/results/M/fold0/splits.json")
EXP21_REF = Path("/home/gchie/workspace/wt/focus8/results/reference/v2026-11-19")
FEAT = "LONG/cv_ft/fold{f}/untrimmed/checkpoint_best_steps/all"
TARGETS = ["Grasping_catheter_tip", "Clip_or_coagulate_vas_deferens_-_right",
           "Identification_and_clipping_of_SV_arteries_-_right", "Metal_clip_on_right_pedicle",
           "Start_dissection", "Incision_of_Denonvilliers_fascia", "Transection_of_the_urethra", "Stitch_in_bladder"]


def target_columns(class_names):
    """Columns of ``["Idle"] + TARGETS`` in a list of class names.

    >>> target_columns(["Idle"] + ["x"] + TARGETS[::-1])[-1]
    2
    """
    return [class_names.index("Idle")] + [class_names.index(t) for t in TARGETS]


def fold_splits(fold):
    import pandas as pd

    d = S / "LONG/cv_ft" / f"fold{fold}"
    train = sorted(set(pd.read_csv(d / "train_data.csv").patient_name))
    inner = sorted(set(pd.read_csv(d / "val_data.csv").patient_name))
    held = sorted(p.stem for p in (S / "TEMPORAL/cv_ft_dilate3" / f"fold{fold}" / "val").glob("*.npz"))
    assert not (set(train) & set(inner)) and not (set(train) & set(held)) and not (set(inner) & set(held))
    return {"train": train, "inner_val": inner, "held_out": held}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", default="results/exp22")
    a = ap.parse_args()
    import pandas as pd

    out = Path(a.out) / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    sp = fold_splits(a.fold)
    if a.fold == 0:
        ref = json.loads(EXP21_SPLITS.read_text())
        assert all(sorted(ref[k]) == sp[k] for k in sp), "fold 0 splits differ from EXP-21"
    feats = sorted(p.stem for p in (S / FEAT.format(f=a.fold)).glob("*.npz"))
    missing = [v for k in sp for v in sp[k] if v not in feats]
    assert not missing, missing
    (out / "splits.json").write_text(json.dumps(sp, indent=1))
    names = list(pd.read_csv(S / "LONG/cv_ft" / f"fold{a.fold}" / "distributions/train_steps_distribution.csv")["name"])
    cols = target_columns(names)
    ev_names = ["Idle"] + TARGETS
    for tag, sub in (("s0", "cv_ft_dilate3"), ("s1", "cv_ft_dilate3_s1")):
        d = out / f"reference_{tag}"
        d.mkdir(exist_ok=True)
        src = S / "TEMPORAL" / sub / f"fold{a.fold}" / "val"
        for v in sp["held_out"]:
            z = dict(np.load(src / f"{v}.npz"))
            z["probs_steps"] = z["probs_steps"][:, cols]
            np.savez_compressed(d / f"{v}.npz", **z)
            if a.fold == 0:
                r = np.load(EXP21_REF / f"dilate3{'' if tag == 's0' else '_s1'}_focus8" / f"{v}.npz")
                assert np.array_equal(r["probs_steps"], z["probs_steps"]) and np.array_equal(r["second"], z["second"])
        (d / "meta.json").write_text(json.dumps({"fps": 4, "event_classes": ev_names, "source": str(src),
                                                 "note": "33-class probs, target columns only"}, indent=1))
    print(f"fold {a.fold}: {len(sp['train'])}/{len(sp['inner_val'])}/{len(sp['held_out'])} videos, {len(feats)} feature files, "
          f"columns {cols}")


if __name__ == "__main__":
    main()
