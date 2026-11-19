#!/usr/bin/env python3
"""
EXP-21: evaluate saved per-row predictions (``<pred-dir>/<RARPxx>.npz`` with ``second``, ``probs_steps``
and/or ``probs_phases``) on the 8 target events and the 15 phases, with the revised labels.

Events (point spotting, ground truth at ``time_s``):
  * class names: ``meta.json`` ``event_classes`` (index 0 = background), else the 9 focus8 classes;
  * decoders: ``nms`` (per-class peaks, 5 s half window), ``caps``, ``caps_chain`` (priors from the target
    events of the training videos only, ``tapis.evaluate.structured_decoding``), ``phase_caps_chain``
    (event scores gated by the smoothed predicted phase, as eval_constraints.py);
  * ground truth inside ignore rows is dropped, detections inside the ignore windows of ``--ignore-dir``
    (the 1 fps label version, as the reference) are dropped;
  * per-class AP and group mAP (short/long targets present in the class list) at +-1/2/5 s, with a
    per-video bootstrap CI (videos resampled with replacement; matching is per video, so the
    per-detection TP flags are computed once).
Phases: row accuracy and macro F1 at the prediction rate; per-video segmental F1@50 (mean over videos) at
1 Hz (4 Hz posteriors averaged per second), raw argmax (P0) and P1 (hysteresis + per-phase minimum
duration, tools/phase_decode.py defaults, duration priors from the training videos).

Aggregates only: nothing per patient is written.

    python tools/eval_focus8.py --pred-dir results/M/fold0/untrimmed --splits results/M/fold0/splits.json \
        --set held_out --out results/M/fold0/eval/test.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "tools", REPO / "detectron2"):
    sys.path.insert(0, str(p))

from tapis.datasets import focus8 as F  # noqa: E402
from tapis.evaluate import spotting_eval as se  # noqa: E402
from tapis.evaluate import structured_decoding as sd  # noqa: E402

DELTAS = (1, 2, 5)
LAB4 = "/home/gchie/workspace/data/orsi_labels_revised/v2026-11-19_4hz"
LAB1 = "/home/gchie/workspace/data/orsi_labels_revised/v2026-11-19"
FRAME_DIR = "/home/gchie/workspace/data/orsi_tensors"
VARIANTS = ("nms", "caps", "caps_chain", "phase_caps_chain")


def labels(label_dir, patient, fps=4):
    return F.VideoLabels(Path(label_dir) / patient / f"{patient}_all_labels.csv", fps=fps)


def gt_events(lab, names):
    """[(class index in ``names``, time)] of the target events outside ignore rows."""
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for t, k in lab.events:
        if k <= 0 or F.TARGETS[k - 1] not in idx:
            continue
        r = min(lab.n - 1, int(round(t * lab.fps)))
        if not lab.ignore[r]:
            out.append((idx[F.TARGETS[k - 1]], float(t)))
    return out


def row_phase(lab, sec):
    return lab.phase[np.clip(np.round(np.asarray(sec) * lab.fps).astype(int), 0, lab.n - 1)]


def priors(train, names, label_dir):
    ev, lengths = {}, {}
    for p in train:
        lab = labels(label_dir, p)
        ev[p] = [(c, t, int(lab.phase[min(lab.n - 1, int(round(t * lab.fps)))])) for c, t in gt_events(lab, names)]
        lengths[p] = lab.n / lab.fps
    return sd.estimate_priors(ev, lengths, len(names), len(F.PHASE_IDS), background_classes=(0,))


def smooth_rows(smooth_s, fps):
    n = int(round(smooth_s * fps))
    return 1 if smooth_s <= 1 else (n if n % 2 else n + 1)


def detect(z, fps, variant, pri, ign, alpha=0.1, smooth_s=61, floor=0.2, nms_window=5.0):
    probs, sec = z["probs_steps"], z["second"]
    if variant.startswith("phase"):
        m = floor + (1.0 - floor) * pri["phase_given_event"]
        probs = sd.phase_gate(probs, sd.smooth(z["probs_phases"], smooth_rows(smooth_s, fps)), m)
    dets = se.decode_events(probs, se.window_frames(nms_window, fps), 0.0, background_classes=(0,))
    if "caps" in variant:
        dets = sd.apply_caps(dets, pri["caps"], alpha)
    if "chain" in variant:
        dets = sd.chain_decode(dets, pri["chain"], alpha)
    return se.drop_ignored([(c, float(sec[t]), s) for c, t, s in dets if s > 0], ign)


def matches(dets, gts, classes):
    """{(video, class, delta): (scores, tp)}, {(video, class): n_gt}."""
    m, n = {}, {}
    for v in gts:
        for c in classes:
            pt = [t for k, t, _ in dets[v] if k == c]
            ps = [s for k, _, s in dets[v] if k == c]
            gt = [t for k, t in gts[v] if k == c]
            n[(v, c)] = len(gt)
            for d in DELTAS:
                tp, _, _ = se.match_events(pt, ps, gt, d)
                m[(v, c, d)] = (np.asarray(ps, float), np.asarray(tp, float))
    return m, n


def ap_table(m, n, videos, classes):
    """ap[d][c] pooled over ``videos`` (a list, repeats allowed)."""
    out = {}
    for d in DELTAS:
        out[d] = {}
        for c in classes:
            ng = sum(n[(v, c)] for v in videos)
            s = np.concatenate([m[(v, c, d)][0] for v in videos])
            tp = np.concatenate([m[(v, c, d)][1] for v in videos])
            out[d][c] = se.average_precision(s, tp, ng)
    return out


def group_map(ap, groups):
    return {g: {d: float(np.nanmean([ap[d][c] for c in cls])) if cls else float("nan") for d in DELTAS}
            for g, cls in groups.items()}


def spotting(dets, gts, names, n_boot, seed):
    classes = list(range(1, len(names)))
    groups = {"short": [c for c in classes if names[c] in F.SHORT], "long": [c for c in classes if names[c] in F.LONG]}
    groups = {g: c for g, c in groups.items() if c}
    m, n = matches(dets, gts, classes)
    videos = sorted(gts)
    ap = ap_table(m, n, videos, classes)
    res = {"per_class_ap": {str(d): {names[c]: ap[d][c] for c in classes} for d in DELTAS},
           "n_gt": {names[c]: int(sum(n[(v, c)] for v in videos)) for c in classes},
           "group_map": {g: {str(d): v for d, v in x.items()} for g, x in group_map(ap, groups).items()}}
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = {g: {d: [] for d in DELTAS} for g in groups}
        cls_boot = {c: {d: [] for d in DELTAS} for c in classes}
        for _ in range(n_boot):
            vs = list(rng.choice(videos, len(videos), replace=True))
            a = ap_table(m, n, vs, classes)
            for g, x in group_map(a, groups).items():
                for d, v in x.items():
                    boots[g][d].append(v)
            for c in classes:
                for d in DELTAS:
                    cls_boot[c][d].append(a[d][c])
        ci = lambda xs: [float(np.nanpercentile(xs, 2.5)), float(np.nanpercentile(xs, 97.5))]
        res["group_map_ci95"] = {g: {str(d): ci(boots[g][d]) for d in DELTAS} for g in groups}
        res["per_class_ap_ci95"] = {str(d): {names[c]: ci(cls_boot[c][d]) for c in classes} for d in DELTAS}
    return res


def per_second(probs, sec):
    s = np.floor(np.asarray(sec) + 1e-6).astype(int)
    out = np.zeros((s.max() + 1, probs.shape[1]))
    np.add.at(out, s, probs)
    cnt = np.bincount(s, minlength=len(out))
    keep = cnt > 0
    return out[keep] / cnt[keep, None], np.flatnonzero(keep)


def phases(preds, labs, train, label_dir):
    import phase_decode as pd_

    C = len(F.PHASE_IDS)
    tr_gt = []
    for p in train:
        lab = labels(label_dir, p)
        tr_gt.append(lab.phase[:: lab.fps])
    pri = pd_.fit_priors(tr_gt, C)
    rows_t, rows_p, f1 = [], [], {"P0": [], "P1": []}
    acc1 = {"P0": [], "P1": []}
    for v, z in preds.items():
        lab = labs[v]
        y = row_phase(lab, z["second"])
        rows_t.append(y)
        rows_p.append(z["probs_phases"].argmax(1))
        p1s, secs = per_second(z["probs_phases"], z["second"])
        g1 = row_phase(lab, secs)
        for name, dec in (("P0", p1s.argmax(1)), ("P1", pd_.decode_hysteresis(p1s, pri))):
            mm = pd_.video_metrics(dec, g1, C)
            f1[name].append(mm["F1@50"])
            acc1[name].append(mm["acc"])
    from sklearn.metrics import f1_score

    t, pr = np.concatenate(rows_t), np.concatenate(rows_p)
    return {"row_accuracy": float((t == pr).mean()), "row_macro_f1": float(f1_score(t, pr, average="macro")),
            "F1@50_mean_video": {k: float(np.mean(x)) for k, x in f1.items()},
            "acc_1hz_mean_video": {k: float(np.mean(x)) for k, x in acc1.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--splits", required=True, help="splits.json (train = priors, --set = evaluated videos)")
    ap.add_argument("--set", default="held_out", choices=["held_out", "inner_val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-dir", default=LAB4)
    ap.add_argument("--ignore-dir", default=LAB1)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--phase-smooth", type=float, default=61)
    ap.add_argument("--gate-floor", type=float, default=0.2)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from eval_untrimmed import ignore_windows

    pred_dir = Path(args.pred_dir)
    meta = json.loads((pred_dir / "meta.json").read_text()) if (pred_dir / "meta.json").exists() else {}
    sp = json.loads(Path(args.splits).read_text())
    videos, train = sp[args.set], sp["train"]
    fps = float(meta.get("fps", 1))
    preds = {v: dict(np.load(pred_dir / f"{v}.npz")) for v in videos}
    labs = {v: labels(args.label_dir, v) for v in videos}
    report = {"pred_dir": str(pred_dir.resolve()), "set": args.set, "n_videos": len(videos), "fps": fps,
              "label_dir": args.label_dir, "ignore_dir": args.ignore_dir, "args": vars(args)}
    has_ev = "probs_steps" in next(iter(preds.values()))
    if has_ev:
        names = meta.get("event_classes") or F.EVENT_CLASSES
        assert preds[videos[0]]["probs_steps"].shape[1] == len(names), "event classes != probs_steps columns"
        pri = priors(train, names, args.label_dir)
        report["event_classes"] = names
        report["prior_caps"] = {names[c]: k for c, k in pri["caps"].items()}
        report["prior_chain"] = [names[c] for c in pri["chain"]]
        ign = ignore_windows(FRAME_DIR, args.ignore_dir, videos)
        gts = {v: gt_events(labs[v], names) for v in videos}
        has_ph = "probs_phases" in preds[videos[0]]
        for var in args.variants:
            if var.startswith("phase") and not has_ph:
                continue
            dets = {v: detect(preds[v], fps, var, pri, ign[v], args.alpha, args.phase_smooth, args.gate_floor)
                    for v in videos}
            report[var] = spotting(dets, gts, names, args.bootstrap, args.seed)
    if "probs_phases" in next(iter(preds.values())):
        report["phases"] = phases(preds, labs, train, args.label_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    for var in args.variants:
        if var in report:
            gm, ci = report[var]["group_map"], report[var].get("group_map_ci95", {})
            print(var, " | ".join(f"{g} " + " ".join(f"{gm[g][str(d)]:.3f}" + (
                f"[{ci[g][str(d)][0]:.2f},{ci[g][str(d)][1]:.2f}]" if ci else "") for d in DELTAS) for g in gm))
    if "phases" in report:
        print("phases", json.dumps(report["phases"]))


if __name__ == "__main__":
    main()
