#!/usr/bin/env python3
"""
Compare structured decoders on saved untrimmed predictions (no retraining).

Priors (event/phase compatibility, per-video cardinality, order of the milestone
events) are estimated on the run's *training* patients only, then applied to the
per-second probabilities written by `tools/infer_untrimmed.py`:

  nms               per-class peak picking (baseline, as in eval_untrimmed.py)
  phase             event scores gated by the model's own (smoothed) phase posterior
  phase_oracle      same gate with the ground-truth phase: upper bound of phase info
  caps              keep the top-`cap` detections of milestone classes per video
  caps_chain        caps + order of single-occurrence milestones (DP)
  phase_caps_chain  all constraints with predicted phases
  oracle_caps_chain all constraints with ground-truth phases

`phase` is causal-compatible if the smoothing is made causal; `caps` and `chain`
look at the whole video (offline).

Example:
    python tools/eval_constraints.py \
        --pred-dir outputs/orsi/LONG/control/totale/untrimmed/checkpoint_best_steps/val \
        --frame-dir /home/gchie/workspace/data/orsi_tensors
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from eval_untrimmed import DELTAS, build_label_table, global_to_local, idle_index, load_cfg
from tapis.evaluate import spotting_eval as se
from tapis.evaluate import structured_decoding as sd

VARIANTS = ("nms", "phase", "phase_oracle", "caps", "caps_chain", "phase_caps_chain", "oracle_caps_chain")


def read_phase_labels(frame_dir, patient, g2l_ph):
    labels = pd.read_csv(Path(frame_dir) / patient / "Label" / f"{patient}_all_labels.csv", usecols=["phase_id"])
    return labels["phase_id"].map(g2l_ph).fillna(0).astype(int).to_numpy()


def events_by_video(table, patients, g2l, g2l_ph):
    events = table[table["event_name"].str.strip().str.lower() != "idle"]
    events = events[events["event_id"].isin(g2l)]
    out = {p: [] for p in patients}
    for row in events.itertuples():
        out[row.patient_name].append((g2l[row.event_id], float(row.second), int(g2l_ph.get(row.phase_id, 0))))
    return out


def prepare(pred_dir, frame_dir, patients=None):
    """
    Everything the decoders need for the saved predictions in ``pred_dir``: priors
    estimated on the run's training patients, ground truth and per-second
    probabilities of the evaluated ``patients`` (default: all of ``meta.json``).
    """
    pred_dir = Path(pred_dir).resolve()
    meta = json.loads((pred_dir / "meta.json").read_text())
    run = Path(meta["run"])
    cfg = load_cfg(run, meta["ckpt"])
    patients = patients or meta["patients"]
    task = "steps" if "steps" in meta["tasks"] else "actions"

    # ---- priors from the training patients -------------------------------------
    train = meta.get("train_patients") or [
        Path(f).name.removesuffix(".csv") for f in cfg.ENDOVIS_DATASET.TRAIN_LISTS
    ]
    work = pred_dir / "constraints"
    work.mkdir(exist_ok=True)
    tr_table, tr_ids = build_label_table(cfg, train, frame_dir, work)
    g2l, names = global_to_local(run, task, tr_ids[task])
    g2l_ph, ph_names = global_to_local(run, "phases", tr_ids["phases"])
    bg = idle_index(names)
    tr_events = events_by_video(tr_table, train, g2l, g2l_ph)
    lengths = {p: len(read_phase_labels(frame_dir, p, g2l_ph)) for p in train}
    priors = sd.estimate_priors(tr_events, lengths, len(names), len(ph_names), background_classes=bg)

    # ---- evaluation patients ----------------------------------------------------
    table, _ = build_label_table(cfg, patients, frame_dir, work)
    ev = events_by_video(table, patients, g2l, g2l_ph)
    preds = {p: dict(np.load(pred_dir / f"{p}.npz")) for p in patients}
    return SimpleNamespace(
        pred_dir=pred_dir,
        meta=meta,
        patients=patients,
        task=task,
        has_phases="phases" in meta["tasks"],
        names=names,
        ph_names=ph_names,
        bg=bg,
        priors=priors,
        gts={p: [(c, t) for c, t, _ in ev[p]] for p in patients},
        preds=preds,
        durations={p: float(len(preds[p]["second"])) for p in patients},
        gt_phases={p: read_phase_labels(frame_dir, p, g2l_ph) for p in patients},
    )


def detections(prep, p, variant, alpha, smooth_s, floor, nms_window=5.0):
    """Decoded ``[(class, second, score)]`` of video ``p`` with one of ``VARIANTS``."""
    probs = prep.preds[p][f"probs_{prep.task}"]
    sec = prep.preds[p]["second"]
    m = floor + (1.0 - floor) * prep.priors["phase_given_event"]
    if "oracle" in variant:
        probs = sd.phase_gate(probs, np.eye(len(prep.ph_names))[prep.gt_phases[p]], m)
    elif variant.startswith("phase"):
        probs = sd.phase_gate(probs, sd.smooth(prep.preds[p]["probs_phases"], smooth_s), m)
    dets = se.decode_events(probs, nms_window, 0.0, background_classes=prep.bg)
    if "caps" in variant:
        dets = sd.apply_caps(dets, prep.priors["caps"], alpha)
    if "chain" in variant:
        dets = sd.chain_decode(dets, prep.priors["chain"], alpha)
    return [(c, float(sec[t]), s) for c, t, s in dets if s > 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--frame-dir", default="/data/orsi_tensors")
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--nms-window", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.0], help="score factor of rejected detections")
    parser.add_argument("--phase-smooth", type=int, nargs="+", default=[61], help="moving average of phase probs (s)")
    parser.add_argument(
        "--gate-floor", type=float, nargs="+", default=[0.0], help="gate = floor + (1 - floor) * compatibility"
    )
    args = parser.parse_args()

    prep = prepare(args.pred_dir, args.frame_dir, args.patients)
    pred_dir, meta, patients, names, bg = prep.pred_dir, prep.meta, prep.patients, prep.names, prep.bg
    gts, durations, has_phases = prep.gts, prep.durations, prep.has_phases

    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    results = {}
    for alpha in args.alpha:
        for smooth_s in args.phase_smooth:
            for floor in args.gate_floor:
                for variant in VARIANTS:
                    uses_gate = "phase" in variant or "oracle" in variant
                    uses_alpha = "caps" in variant or "chain" in variant
                    if (variant.startswith("phase") and not has_phases) or (
                        # skip grid points that do not change the variant
                        (not uses_gate and (smooth_s, floor) != (args.phase_smooth[0], args.gate_floor[0]))
                        or ("oracle" in variant and smooth_s != args.phase_smooth[0])
                        or (not uses_alpha and alpha != args.alpha[0])
                    ):
                        continue
                    key = f"{variant}|a={alpha:g}|s={smooth_s}|f={floor:g}"
                    dets = {
                        p: detections(prep, p, variant, alpha, smooth_s, floor, args.nms_window) for p in patients
                    }
                    res = se.spotting_map(dets, gts, DELTAS, num_classes=len(names), ignore_classes=bg)
                    op = se.best_f1_threshold(dets, gts, 5, durations, thresholds, ignore_classes=bg)
                    results[key] = {
                        "map": {str(d): res["map"][d] for d in DELTAS},
                        "avg_map_tight": se.average_map(res["map"], se.TIGHT_DELTAS),
                        "avg_map_loose": se.average_map(res["map"], se.LOOSE_DELTAS),
                        "best_f1_5s": op,
                        "per_class_ap_5s": {names[c]: a for c, a in res["ap"][5].items()},
                    }

    report = {
        "meta": meta,
        "patients": patients,
        "args": vars(args),
        "caps": {names[c]: k for c, k in prep.priors["caps"].items()},
        "chain": [names[c] for c in prep.priors["chain"]],
        "results": results,
    }
    tag = "" if patients == meta["patients"] else "_" + "-".join(p.removeprefix("RARP") for p in patients)
    out = pred_dir / f"constraints{tag}_nms{args.nms_window:g}.json"
    out.write_text(json.dumps(report, indent=2, default=float))

    print(f"chain ({len(report['chain'])}): {report['chain']}")
    print(f"caps: {report['caps']}")
    header = f"{'variant':38s} {'mAP@1':>6s} {'@2':>6s} {'@5':>6s} {'@10':>6s} {'@30':>6s} {'tight':>6s} {'loose':>6s} | {'F1@5':>5s} {'P':>5s} {'R':>5s} {'FA/h':>6s}"
    print(header)
    for v, r in results.items():
        mm = r["map"]
        op = r["best_f1_5s"]
        print(
            f"{v:38s} {mm['1']:6.3f} {mm['2']:6.3f} {mm['5']:6.3f} {mm['10']:6.3f} {mm['30']:6.3f} "
            f"{r['avg_map_tight']:6.3f} {r['avg_map_loose']:6.3f} | {op['f1']:5.3f} {op['precision']:5.3f} "
            f"{op['recall']:5.3f} {op['false_alarms_per_hour']:6.1f}"
        )
    print(f"Full report: {out}")


if __name__ == "__main__":
    main()
