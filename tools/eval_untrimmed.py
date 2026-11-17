#!/usr/bin/env python3
"""
Evaluate per-second predictions from `tools/infer_untrimmed.py` on full videos.

Ground truth is built with the training dataset class itself (`Orsi._load_data`:
same exclusions, same event collapsing) and mapped to the model's output indices
with the run's `distributions/train_*_distribution.csv`, so classes are aligned
with training by construction.

Reported (aggregate only, nothing per patient leaves the json):
  * sanity check: frame-wise mAP on the trimmed validation rows (event frames +
    ADD_IDLE frames), which must match the mAP logged during training;
  * steps as point events: spotting mAP@delta (+/- seconds), tight/loose
    average-mAP, per-class AP, and operating points (precision, recall, F1,
    false alarms per hour, duplicates, timing error) for two decoders:
    `nms` (per-class peak picking) and `argmax` (naive frame-wise classifier);
  * phases as dense labels: frame accuracy, macro F1 and frame-wise mAP.

Options for other label sets and frame rates (defaults reproduce the historical evaluation):
  * ``--label-dir``: ground truth from ``<label-dir>/RARPxx/RARPxx_all_labels.csv`` (e.g. the
    revised labels of `build_revised_labels.py`); default: the run's DATA.LABEL_DIR, else the
    original CSVs. Rows with ``ignore=1`` drop their ground-truth events, and detections falling
    in an ignore window are discarded (for every decoder);
  * ``--fps``: rows per second of the predictions (default: ``meta.json`` ``fps``, else 1). All
    tolerances and the NMS window stay in seconds;
  * ``--fractional-gt``: ground-truth time = the ``time_s`` column instead of the row's second;
  * ``--ignore-dir``: ignore windows from another label dir (e.g. the 1 fps one for 4 fps runs,
    so every arm drops the same detections);
  * ``--classes NAME:c1,c2,...``: extra metrics restricted to a class subset (repeatable);
  * ``--report-dir`` / ``--tag``: where the report (and the label-table scratch files) go.

Example:
    python tools/eval_untrimmed.py \
        --pred-dir outputs/orsi/LONG/control/totale/untrimmed/checkpoint_best_steps/val \
        --frame-dir /home/gchie/workspace/data/orsi_tensors
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "detectron2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch
from fvcore.common.config import CfgNode
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize

from tapis.config.defaults import get_cfg
from tapis.datasets.orsi import Orsi
from tapis.evaluate import spotting_eval as se

DELTAS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30)
TASK_COLUMN = {"steps": "event", "actions": "event", "phases": "phase", "classes": "classes"}


def load_cfg(run, ckpt):
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # older checkpoints carry keys removed from defaults.py
    cfg.merge_from_other_cfg(CfgNode.load_cfg(torch.load(run / ckpt, map_location="cpu", weights_only=False)["cfg"]))
    return cfg


def build_label_table(cfg, patients, frame_dir, out_dir, label_dir=None):
    """
    Rows the training pipeline would use for `patients`, with *global* ids after collapsing.

    `Orsi.collapse_event_dfs` only collapses an event if its target event occurs in
    the loaded data, so the result depends on which patients are loaded (a single
    test video can end up with 36 classes instead of 33). Loading the training
    patients together with `patients` reproduces the collapsing seen in training;
    the table is then restricted to `patients`. ADD_IDLE picks rows per patient with
    a fixed seed, so the Idle rows of `patients` are the same as in a split of their own.
    """
    cfg.defrost()
    if label_dir is not None:
        cfg.DATA.LABEL_DIR = label_dir
    cfg.ENDOVIS_DATASET.FRAME_DIR = frame_dir
    if "ORSI_ROOT_DIR" in cfg.ENDOVIS_DATASET:
        cfg.ENDOVIS_DATASET.ORSI_ROOT_DIR = frame_dir
    cfg.OUTPUT_DIR = str(out_dir)
    train = [Path(f).name.removesuffix(".csv") for f in cfg.ENDOVIS_DATASET.TRAIN_LISTS]
    cfg.ENDOVIS_DATASET.VAL_LISTS = [f"{p}.csv" for p in dict.fromkeys(train + list(patients))]
    with contextlib.redirect_stdout(io.StringIO()):
        ds = Orsi(cfg, "val", load=False)
        ds._load_data()
        ds.generate_weight_vector()  # class order the model was trained with (writes to OUTPUT_DIR)
    table = ds.filtered_dfs if ds.filtered_dfs is not None else ds.dfs
    table = table[table["patient_name"].isin(patients)].reset_index(drop=True)
    return table, {task: list(ds.counter[task]["id"]) for task in ds.counter}


def global_to_local(run, task, split_ids):
    """Global id -> model output index, from the run's saved class distribution."""
    dist_dir = run / "distributions"
    path = dist_dir / f"train_{task}_distribution.csv"
    if not path.exists():
        path = dist_dir / f"{task}_distribution.csv"
    dist = pd.read_csv(path, skipinitialspace=True)  # older runs pad the columns with spaces
    dist.columns = [c.strip() for c in dist.columns]
    if list(dist["id"]) != list(split_ids):
        raise RuntimeError(f"{task}: class order of {path.name} differs from the one rebuilt on this split")
    return dict(zip(dist["id"], range(len(dist)))), [str(n).strip() for n in dist["name"]]


def load_labels(frame_dir, label_dir, patient):
    """Label CSV of ``patient``: ``<label_dir>/P/P_all_labels.csv`` if ``label_dir``, else the original."""
    if label_dir:
        return pd.read_csv(Path(label_dir) / patient / f"{patient}_all_labels.csv")
    return pd.read_csv(Path(frame_dir) / patient / "Label" / f"{patient}_all_labels.csv")


def label_dir_of(cfg, label_dir=None, meta=None):
    """
    ``--label-dir`` if given, else the predictions' ``meta.json`` ``label_dir`` (written for grid
    runs), else the run's DATA.LABEL_DIR ("" = original labels).
    """
    if label_dir is not None:
        return label_dir
    if meta and meta.get("label_dir"):
        return meta["label_dir"]
    return getattr(cfg.DATA, "LABEL_DIR", "") if "DATA" in cfg else ""


def ignore_windows(frame_dir, label_dir, patients):
    """
    ``{patient: [(lo, hi), ...]}`` ignore intervals (s) from the ``ignore`` column of the CSVs in
    ``label_dir`` (none without it); the row spacing (1 or 0.25 s) is read from the CSV, so a 1 fps
    label dir gives the same windows to 1 fps and 4 fps predictions (``--ignore-dir``).
    """
    out = {}
    for p in patients:
        lab = load_labels(frame_dir, label_dir, p) if label_dir else None
        if lab is None or "ignore" not in lab:
            out[p] = []
            continue
        sec = np.unique(lab["second"].to_numpy(dtype=float))
        rate = 1.0 / float(np.median(np.diff(sec))) if len(sec) > 1 else 1.0
        times = lab.loc[lab["ignore"].fillna(0).astype(int) == 1, "second"].drop_duplicates()
        out[p] = se.ignore_intervals(times.to_numpy(), rate)
    return out


def gt_time(row, fractional):
    t = getattr(row, "time_s", None) if fractional else None
    return float(row.second if t is None or pd.isna(t) else t)


def parse_class_groups(specs, names):
    """
    ``["short:A,B", "C"]`` -> ``{"short": [iA, iB], "C": [iC]}`` (output indices; unknown names fail).
    """
    out = {}
    idx = {n: i for i, n in enumerate(names)}
    for spec in specs or []:
        key, _, lst = spec.rpartition(":") if ":" in spec else (spec, "", spec)
        cls = [c.strip() for c in lst.split(",") if c.strip()]
        missing = [c for c in cls if c not in idx]
        if missing:
            raise ValueError(f"unknown classes {missing}")
        out[key or "subset"] = [idx[c] for c in cls]
    return out


def subset_metrics(dets, gts, groups, names, bg, durations, deltas=DELTAS):
    """Spotting mAP / per-class AP / best F1@5 s restricted to each class group."""
    out = {}
    for key, cls in groups.items():
        ign = tuple(c for c in range(len(names)) if c not in cls or c in bg)
        res = se.spotting_map(dets, gts, deltas, num_classes=len(names), ignore_classes=ign)
        op = se.best_f1_threshold(dets, gts, 5, durations, np.round(np.arange(0.05, 1.0, 0.05), 2), ignore_classes=ign)
        out[key] = {
            "classes": [names[c] for c in cls],
            "map": {str(d): v for d, v in res["map"].items()},
            "avg_map_tight_1_5s": (
                se.average_map(res["map"], se.TIGHT_DELTAS) if set(se.TIGHT_DELTAS) <= set(res["map"]) else None
            ),
            "per_class_ap": {str(d): {names[c]: res["ap"][d][c] for c in res["ap"][d]} for d in (1, 2, 5) if d in res["ap"]},
            "n_gt": {names[c]: n for c, n in res["n_gt"].items()},
            "best_f1_5s": op,
        }
    return out


def phase_rows(frame_dir, label_dir, patient, fps=1):
    """
    Rows aligned with the predictions: at 1 fps the original CSV (the frame list inference used),
    above 1 fps the unique frame_id rows of the ``label_dir`` grid CSV.
    """
    if fps == 1:
        return pd.read_csv(Path(frame_dir) / patient / "Label" / f"{patient}_all_labels.csv")
    return load_labels(frame_dir, label_dir, patient).drop_duplicates("frame_id").reset_index(drop=True)


def idle_index(names):
    """Output indices that mean "no event" (runs trained without Idle have none)."""
    return tuple(i for i, n in enumerate(names) if n.lower() == "idle")


def frame_map(y_true, probs):
    """Mean AP over classes that have at least one positive (one-vs-rest)."""
    aps = []
    for c in range(probs.shape[1]):
        pos = y_true == c
        if pos.any():
            aps.append(average_precision_score(pos, probs[:, c]))
    return float(np.mean(aps))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--frame-dir", default="/data/orsi_tensors")
    parser.add_argument("--nms-window", type=float, default=5.0, help="NMS half window (s)")
    parser.add_argument("--op-deltas", type=float, nargs="*", default=[2, 5])
    parser.add_argument("--patients", nargs="*", help="Evaluate only these patients (default: all predicted)")
    parser.add_argument("--label-dir", help="revised labels <dir>/RARPxx/RARPxx_all_labels.csv (default: run's)")
    parser.add_argument("--fps", type=float, help="rows per second of the predictions (default: meta.json, else 1)")
    parser.add_argument("--fractional-gt", action="store_true", help="ground truth at time_s, not at the row second")
    parser.add_argument("--classes", nargs="*", help="NAME:class1,class2,... extra metrics on class subsets")
    parser.add_argument("--ignore-dir", help="take the ignore windows from this label dir (default: --label-dir)")
    parser.add_argument("--report-dir", help="default: --pred-dir")
    parser.add_argument("--tag", default="", help="suffix of the report file name")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    meta = json.loads((pred_dir / "meta.json").read_text())
    run = Path(meta["run"])
    cfg = load_cfg(run, meta["ckpt"])
    patients = args.patients or meta["patients"]
    missing = set(patients) - set(meta["patients"])
    if missing:
        raise ValueError(f"No predictions for {sorted(missing)}")
    fps = args.fps or meta.get("fps", 1)
    label_dir = label_dir_of(cfg, args.label_dir, meta)
    report_dir = Path(args.report_dir).resolve() if args.report_dir else pred_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    table, split_ids = build_label_table(cfg, patients, args.frame_dir, report_dir, label_dir)
    report = {"meta": meta, "patients": patients, "nms_window": args.nms_window}
    if label_dir or fps != 1 or args.fractional_gt:  # absent keys = historical evaluation
        report.update(label_dir=label_dir, fps=fps, fractional_gt=args.fractional_gt)

    preds = {p: dict(np.load(pred_dir / f"{p}.npz")) for p in patients}
    durations = {p: float(len(preds[p]["second"])) / fps for p in patients}
    ign = ignore_windows(args.frame_dir, args.ignore_dir or label_dir, patients)
    report["hours"] = sum(durations.values()) / 3600

    # ---- steps as point events -------------------------------------------------
    task = "steps" if "steps" in meta["tasks"] else "actions"
    g2l, names = global_to_local(run, task, split_ids[task])
    bg = idle_index(names)
    report["background_classes"] = [names[i] for i in bg]
    events = table[table["event_name"].str.strip().str.lower() != "idle"]
    unmapped = events[~events["event_id"].isin(g2l)]
    gts = {p: [] for p in patients}
    n_gt_ignored = 0
    for row in events[events["event_id"].isin(g2l)].itertuples():
        if int(getattr(row, "ignore", 0) or 0) == 1:
            n_gt_ignored += 1
            continue
        gts[row.patient_name].append((g2l[row.event_id], gt_time(row, args.fractional_gt)))
    report["gt_events"] = int(sum(len(v) for v in gts.values()))
    report["gt_events_unmapped"] = int(len(unmapped))
    if any(ign.values()) or n_gt_ignored:
        report["ignore"] = {"gt_events_dropped": n_gt_ignored, "windows": int(sum(len(v) for v in ign.values()))}
    groups = parse_class_groups(args.classes, names)

    # sanity check: the trimmed rows used for validation during training
    trimmed_true, trimmed_probs = [], []
    for row in table.itertuples():
        sec = preds[row.patient_name]["second"]
        i = int(np.searchsorted(sec, row.second))
        trimmed_true.append(g2l.get(row.event_id, 0))
        trimmed_probs.append(preds[row.patient_name][f"probs_{task}"][i])
    trimmed_true, trimmed_probs = np.array(trimmed_true), np.stack(trimmed_probs)
    report["trimmed_check"] = {
        "n_rows": len(trimmed_true),
        "frame_map": frame_map(trimmed_true, trimmed_probs),
        # identical to classification_eval.eval_classification (classes without positives count)
        "frame_map_as_training": float(
            average_precision_score(label_binarize(trimmed_true, classes=list(range(len(names)))), trimmed_probs)
        ),
        "macro_f1": float(f1_score(trimmed_true, trimmed_probs.argmax(1), average="macro")),
    }

    thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    for mode in ("nms", "argmax"):
        dets = {
            p: se.drop_ignored(
                [
                    (c, float(preds[p]["second"][t]), s)
                    for c, t, s in se.decode_events(
                        preds[p][f"probs_{task}"], se.window_frames(args.nms_window, fps), 0.0,
                        background_classes=bg, mode=mode,
                    )
                ],
                ign[p],
            )
            for p in patients
        }
        res = se.spotting_map(dets, gts, DELTAS, num_classes=len(names), ignore_classes=bg)
        ops = {}
        for d in args.op_deltas:
            ops[f"delta_{d:g}"] = {
                "best_f1": se.best_f1_threshold(dets, gts, d, durations, thresholds, ignore_classes=bg),
                "threshold_0.5": se.operating_point(dets, gts, d, durations, 0.5, ignore_classes=bg),
                "all_detections": se.operating_point(dets, gts, d, durations, 0.0, ignore_classes=bg),
            }
        report[f"spotting_{mode}"] = {
            "map": {str(d): v for d, v in res["map"].items()},
            "avg_map_tight_1_5s": se.average_map(res["map"], se.TIGHT_DELTAS),
            "avg_map_loose_5_30s": se.average_map(res["map"], se.LOOSE_DELTAS),
            "per_class_ap_5s": {names[c]: res["ap"][5][c] for c in res["ap"][5]},
            "per_class_ap_30s": {names[c]: res["ap"][30][c] for c in res["ap"][30]},
            "n_gt": {names[c]: n for c, n in res["n_gt"].items()},
            "operating_points": ops,
        }
        if groups:
            report[f"spotting_{mode}"]["groups"] = subset_metrics(dets, gts, groups, names, bg, durations)

    # ---- phases as dense per-second labels --------------------------------------
    if "phases" in meta["tasks"]:
        g2l_ph, ph_names = global_to_local(run, "phases", split_ids["phases"])
        y_true, probs = [], []
        for p in patients:
            labels = phase_rows(args.frame_dir, label_dir, p, fps)
            y_true.append(labels["phase_id"].map(g2l_ph).fillna(0).astype(int).to_numpy())
            probs.append(preds[p]["probs_phases"])
        y_true, probs = np.concatenate(y_true), np.concatenate(probs)
        report["phases_untrimmed"] = {
            "n_seconds": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, probs.argmax(1))),
            "macro_f1": float(f1_score(y_true, probs.argmax(1), average="macro")),
            "frame_map": frame_map(y_true, probs),
        }

    tag = "" if patients == meta["patients"] else "_" + "-".join(p.removeprefix("RARP") for p in patients)
    out = report_dir / f"report{tag}{args.tag}_nms{args.nms_window:g}.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    summary = {
        "trimmed_check": report["trimmed_check"],
        "gt_events": report["gt_events"],
        "gt_events_unmapped": report["gt_events_unmapped"],
        "hours": round(report["hours"], 2),
    }
    for mode in ("nms", "argmax"):
        s = report[f"spotting_{mode}"]
        summary[mode] = {
            "mAP@1/2/5/10/30": [round(s["map"][str(d)], 3) for d in (1, 2, 5, 10, 30)],
            "avg_mAP_tight": round(s["avg_map_tight_1_5s"], 3),
            "avg_mAP_loose": round(s["avg_map_loose_5_30s"], 3),
            "best_f1@5s": {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in s["operating_points"]["delta_5"]["best_f1"].items()
            },
            "all_dets@5s_FA/h": round(s["operating_points"]["delta_5"]["all_detections"]["false_alarms_per_hour"], 1),
        }
        for key, g in s.get("groups", {}).items():
            summary[mode][f"group_{key}_mAP@1/2/5"] = [round(g["map"][str(d)], 3) for d in (1, 2, 5)]
    if "phases_untrimmed" in report:
        summary["phases_untrimmed"] = {k: round(v, 3) for k, v in report["phases_untrimmed"].items()}
    print(json.dumps(summary, indent=2))
    print(f"Full report: {out}")


if __name__ == "__main__":
    main()
