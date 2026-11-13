#!/usr/bin/env python3
"""
Export the per-second phase posteriors of a cross-validated train_hybrid.py run to the shared
posterior format (Esp6):

    <out>/phase_posteriors/<video>.npz   probs float32 [T, C] (softmax, 1 fps), gt int16 [T]
                                         (local phase ids, -1 = unlabelled), fold int
    <out>/phase_posteriors/classes.json  {local id: name}

``--run`` is one seed of a CV group (``<run>/fold*/val/<video>.npz`` with ``probs_phases``, as
written by train_hybrid.py); every video comes from the fold where it was held out. Ground truth
is read from ``<frame-dir>/<video>/Label/<video>_all_labels.csv`` (``phase_id``) and mapped to local
ids with the class order of each fold's encoder run (``distributions/train_phases_distribution.csv``).
Unlike ``eval_constraints.read_phase_labels`` (``fillna(0)``: unmapped ids become Idle), unmapped
ids become -1 here and are counted.

Example:
    python tools/export_phase_posteriors.py --run outputs/orsi/EXP2/F_base/seed0 \
        --out outputs/orsi/EXP6/F_base/seed0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def class_order(run):
    """``(global ids, names)`` of the phase classes of an encoder run, in output order."""
    dist_dir = Path(run) / "distributions"
    path = dist_dir / "train_phases_distribution.csv"
    if not path.exists():
        path = dist_dir / "phases_distribution.csv"
    dist = pd.read_csv(path, skipinitialspace=True)
    dist.columns = [c.strip() for c in dist.columns]
    return [int(i) for i in dist["id"]], [str(n).strip() for n in dist["name"]]


def local_phase_labels(frame_dir, video, global_ids):
    """Per-row local phase ids of a video (-1 for ids outside ``global_ids`` or missing) and the -1 count."""
    lab = pd.read_csv(Path(frame_dir) / video / "Label" / f"{video}_all_labels.csv", usecols=["phase_id"])
    g2l = {g: i for i, g in enumerate(global_ids)}
    loc = lab["phase_id"].map(g2l)
    return loc.fillna(-1).astype(np.int16).to_numpy(), int(loc.isna().sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="CV run dir with fold*/val/*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    args = ap.parse_args()
    out = Path(args.out) / "phase_posteriors"
    out.mkdir(parents=True, exist_ok=True)
    names_ref, n_unmapped, n_rows, n_videos = None, 0, 0, 0
    for fold_dir in sorted(Path(args.run).glob("fold*")):
        val = fold_dir / "val"
        meta = json.loads((val / "meta.json").read_text())
        fold = int(meta["cv_fold"].split("/")[0]) if meta.get("cv_fold") else int(fold_dir.name.removeprefix("fold"))
        ids, names = class_order(meta["run"])
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise RuntimeError(f"{fold_dir}: phase class order differs from the other folds")
        for video in meta["patients"]:
            z = np.load(val / f"{video}.npz")
            if "probs_phases" not in z.files:
                raise RuntimeError(f"{val / video}.npz has no probs_phases: regenerate it by inference first")
            probs = z["probs_phases"].astype(np.float32)
            if probs.shape[1] != len(names):
                raise RuntimeError(f"{video}: {probs.shape[1]} phase outputs, {len(names)} classes")
            gt, miss = local_phase_labels(args.frame_dir, video, ids)
            if len(gt) != len(probs):
                raise RuntimeError(f"{video}: {len(gt)} label rows vs {len(probs)} predicted seconds")
            if not np.allclose(probs.sum(1), 1.0, atol=1e-3):
                raise RuntimeError(f"{video}: phase probabilities do not sum to 1")
            np.savez_compressed(out / f"{video}.npz", probs=probs, gt=gt, fold=np.int64(fold))
            n_unmapped += miss
            n_rows += len(gt)
            n_videos += 1
    (out / "classes.json").write_text(json.dumps({str(i): n for i, n in enumerate(names_ref)}, indent=1))
    print(f"{n_videos} videos, {n_rows} seconds, {len(names_ref)} classes, {n_unmapped} unmapped label rows -> {out}")


if __name__ == "__main__":
    main()
