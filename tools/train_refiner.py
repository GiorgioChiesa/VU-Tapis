#!/usr/bin/env python3
"""
Experiment 5: a local refiner that moves each F_base detection to the onset inside a +/- 6 s window.

F_base (`EXP2/F_base`, G2 MS-TCN on the per-fold fine-tuned features) proposes detections
``(class, t, score)`` (`spotting_eval.decode_events`, NMS 5 s). For every detection with score
>= ``--min-score`` a window of positions ``t + delta`` is read from clip features, a small 1-D CNN
scores every position for every class, the detection's class column is softmaxed over the window
and the detection moves to the argmax (the score is kept). Below a confidence threshold, chosen on
held-out train videos, the time is left unchanged.

Arms (``--arm``):
  R1       features of `tools/extract_shifted_features.py` at 4 positions/s, delta in [-6, 6] step
           0.25 s (49 positions);
  R1-1fps  the existing 1 fps features (`infer_untrimmed.py --features`), delta in -6..6 (13 positions).

Training windows (train videos of the fold only): for every event onset ``t`` (fractional time of
`tools/event_times.py`) ``--jitters`` centers ``c = t + U(-5, 5)`` s (as F_base's errors), snapped to
the arm's grid; target = gaussian (sigma ``--sigma`` s) around ``t`` over the window positions,
normalised; loss = cross-entropy with the softmax over the window. ~15 % of the train videos are an
inner validation set for the epoch and the confidence threshold. Features are z-scored with the
statistics of the training windows.

Stages:
  --stage positions  writes ``<exp>/positions/fold<i>/<patient>.npy`` (sorted unique positions, s) for
                     the R1 extraction: train windows of the fold's train videos and the windows of the
                     F_base detections (score >= --min-score, every --seeds) of its val videos; plus
                     ``<exp>/windows/fold<i>.json`` (the train windows, reused by --stage train).
  --stage train      trains the refiner of --arm with --seed and refines the detections of
                     ``F_base/seed<seed>/fold<i>``; writes ``<exp>/<arm>/seed<s>/fold<i>/val/detections.json``,
                     ``refiner.pt`` and ``train_log.json``; also ``<exp>/R0/...`` (identity: every time kept).

Example:
    python tools/train_refiner.py --stage positions --fold 0/5
    python tools/train_refiner.py --stage train --arm R1 --seed 0 --fold 0/5 --gpu 2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_untrimmed import idle_index
from train_hybrid import cv_split
from tapis.evaluate import spotting_eval as se

ROOT = Path(__file__).resolve().parents[1] / "outputs/orsi"
EXP = ROOT / "EXP5"
HALF = 6.0  # window half width (s)
ARMS = {"R1": 4, "R1-1fps": 1}  # positions per second


def window_offsets(per_second, half=HALF):
    """
    Offsets (s) of the window positions.

    >>> window_offsets(1).tolist()
    [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> len(window_offsets(4)), window_offsets(4)[:3].tolist()
    (49, [-6.0, -5.75, -5.5])
    """
    n = int(round(half * per_second))
    return np.arange(-n, n + 1) / per_second


def snap(t, per_second):
    """
    Nearest grid point.

    >>> snap(3.4, 4), snap(3.4, 1), snap(-0.2, 4)
    (3.5, 3.0, -0.25)
    """
    return float(np.round(t * per_second) / per_second)


def gauss_target(positions, t, sigma):
    """
    Normalised gaussian over the window positions around the true time ``t``.

    >>> g = gauss_target(np.array([0.0, 0.25, 0.5, 0.75]), 0.25, 0.5)
    >>> int(g.argmax()), round(float(g.sum()), 6)
    (1, 1.0)
    """
    g = np.exp(-((positions - t) ** 2) / (2 * sigma**2))
    return g / g.sum()


def load_event_times(patients, names):
    """``{patient: [(class, second, time)]}`` from `tools/event_times.py` (same class vocabulary)."""
    out = {}
    for p in patients:
        ev = json.loads((EXP / "event_times" / f"{p}.json").read_text())["events"]
        assert all(names[e["class"]] == e["name"] for e in ev), f"{p}: event_times built with another vocabulary"
        out[p] = [(e["class"], e["second"], e["time"]) for e in ev]
    return out


def fbase_detections(seed, fold_i, bg):
    """``({patient: [(class, t, score)]}, {patient: rows}, meta)`` of an F_base run (every score)."""
    val = ROOT / "EXP2/F_base" / f"seed{seed}" / f"fold{fold_i}" / "val"
    meta = json.loads((val / "meta.json").read_text())
    dets, lengths = {}, {}
    for p in meta["patients"]:
        z = np.load(val / f"{p}.npz")
        dets[p] = se.decode_events(z["probs_steps"], 5.0, 0.0, background_classes=bg)
        lengths[p] = len(z["second"])
    return dets, lengths, meta


def fold_meta(fold):
    """Split, background classes and number of classes of a fold (from F_base seed 0)."""
    i = int(fold.split("/")[0])
    val = ROOT / "EXP2/F_base/seed0" / f"fold{i}" / "val"
    meta = json.loads((val / "meta.json").read_text())
    names = class_names(Path(meta["run"]))
    bg = idle_index(names)
    split = cv_split(meta["train_patients"] + meta["patients"], fold)
    assert sorted(split["val"]) == sorted(meta["patients"]), "fold split differs from F_base's"
    return i, split, bg, names


def class_names(run):
    import pandas as pd

    dist = pd.read_csv(run / "distributions" / "train_steps_distribution.csv", skipinitialspace=True)
    dist.columns = [c.strip() for c in dist.columns]
    return [str(n).strip() for n in dist["name"]]


def n_seconds(patient):
    import pandas as pd

    lab = pd.read_csv(f"/home/gchie/workspace/data/orsi_tensors/{patient}/Label/{patient}_all_labels.csv")
    return int(lab["second"].max()) + 1


def train_windows(events, patients, jitters, seed=1234):
    """
    ``[(patient, class, time, jitter)]``: ``jitters`` uniform(-5, 5) s jitters per onset (fixed seed,
    shared by every arm and refiner seed).
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in patients:
        for c, _, t in events[p]:
            for j in rng.uniform(-5, 5, size=jitters):
                out.append((p, int(c), float(t), float(j)))
    return out


def window_positions(center, per_second, n_sec):
    """Window positions (s) around ``center``, clamped to the video."""
    return np.clip(center + window_offsets(per_second), 0, n_sec - 1)


# ---- positions stage ---------------------------------------------------------------
def stage_positions(args):
    fi, split, bg, names = fold_meta(args.fold)
    events = load_event_times(split["train"] + split["val"], names)
    wins = train_windows(events, split["train"], args.jitters)
    (EXP / "windows").mkdir(parents=True, exist_ok=True)
    (EXP / "windows" / f"fold{fi}.json").write_text(json.dumps({"fold": args.fold, "windows": wins}))
    pos = {p: set() for p in split["train"] + split["val"]}
    nsec = {p: n_seconds(p) for p in pos}
    for p, c, t, j in wins:
        pos[p].update(np.round(window_positions(snap(t + j, 4), 4, nsec[p]) * 4).astype(int).tolist())
    n_val_windows = 0
    for seed in args.seeds:
        dets, _, _ = fbase_detections(seed, fi, bg)
        for p, dd in dets.items():
            for c, t, s in dd:
                if s >= args.min_score:
                    n_val_windows += 1
                    pos[p].update(np.round(window_positions(float(t), 4, nsec[p]) * 4).astype(int).tolist())
    out = EXP / "positions" / f"fold{fi}"
    out.mkdir(parents=True, exist_ok=True)
    for p, g in pos.items():
        np.save(out / f"{p}.npy", np.array(sorted(g), dtype=np.float64) / 4)
    summary = {
        "fold": args.fold,
        "train_videos": len(split["train"]),
        "val_videos": len(split["val"]),
        "train_windows": len(wins),
        "val_windows": n_val_windows,
        "positions_train_videos": sum(len(pos[p]) for p in split["train"]),
        "positions_val_videos": sum(len(pos[p]) for p in split["val"]),
        "seconds_train_videos": sum(nsec[p] for p in split["train"]),
        "seconds_val_videos": sum(nsec[p] for p in split["val"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ---- features ----------------------------------------------------------------------
class FeatureBank:
    """Feature of any grid position of a video (R1: shifted extraction; R1-1fps: 1 fps features by second)."""

    def __init__(self, arm, fold_i, patients):
        self.per_second = ARMS[arm]
        self.rows, self.feats = {}, {}
        for p in patients:
            if arm == "R1":
                z = np.load(EXP / "feat" / f"fold{fold_i}" / f"{p}.npz")
                keys = np.round(z["positions"] * 4).astype(int)
            else:
                z = np.load(ROOT / "LONG/cv_ft" / f"fold{fold_i}" / "untrimmed/checkpoint_best_steps/all" / f"{p}.npz")
                # one row per CSV row: seconds shared by two events appear twice (same clip), keep the first
                keys = z["second"].astype(int)
            first = {}
            for i, k in enumerate(keys.tolist()):
                first.setdefault(k, i)
            self.rows[p] = first
            self.feats[p] = z["features"]

    def window(self, p, positions):
        k = np.round(positions * self.per_second).astype(int)
        return self.feats[p][[self.rows[p][int(x)] for x in k]]


# ---- model -------------------------------------------------------------------------
class Refiner(nn.Module):
    """3 dilated Conv1d (kernel 5, dilations 1/2/4) -> per-position logits for every class."""

    def __init__(self, d_in, n_classes, ch=128, k=5, dilations=(1, 2, 4), dropout=0.3, feat_dropout=0.2):
        super().__init__()
        self.inp = nn.Dropout(feat_dropout)
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_in if i == 0 else ch, ch, k, padding=d * (k // 2), dilation=d) for i, d in enumerate(dilations)]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(ch, n_classes, 1)

    def forward(self, x, cls):
        """x [B, L, D], cls [B] -> log-softmax over the L positions of the class column [B, L]."""
        h = self.inp(x).transpose(1, 2)
        for i, conv in enumerate(self.convs):
            y = self.drop(F.relu(conv(h)))
            h = y if i == 0 else h + y
        logits = self.head(h)  # B, C, L
        return F.log_softmax(logits[torch.arange(len(cls)), cls], dim=1)


def build_train_set(bank, wins, nsec, sigma):
    xs, cs, ys, info = [], [], [], []
    for p, c, t, j in wins:
        center = snap(t + j, bank.per_second)
        pos = window_positions(center, bank.per_second, nsec[p])
        if not pos.min() <= t <= pos.max():
            continue
        xs.append(bank.window(p, pos))
        cs.append(c)
        ys.append(gauss_target(pos, t, sigma))
        info.append((p, t, center, pos))
    return np.stack(xs), np.array(cs), np.stack(ys).astype(np.float32), info


@torch.no_grad()
def predict(model, x, c, device, bs=1024):
    model.eval()
    out = []
    for s in range(0, len(x), bs):
        out.append(model(torch.from_numpy(x[s : s + bs]).to(device), torch.from_numpy(c[s : s + bs]).to(device)).exp().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


def refine_times(prob, pos, t_hat, thr):
    """
    New time per window: argmax position if its probability >= ``thr``, else ``t_hat``.

    >>> refine_times(np.array([[0.1, 0.8, 0.1], [0.4, 0.3, 0.3]]), np.array([[1, 2, 3], [4, 5, 6.]]),
    ...              np.array([2.5, 5.0]), 0.5).tolist()
    [2.0, 5.0]
    """
    a = prob.argmax(1)
    conf = prob[np.arange(len(prob)), a]
    return np.where(conf >= thr, pos[np.arange(len(pos)), a], t_hat)


def hit_score(err):
    """Mean of the shares within 1 s and within 2 s (what mAP@1-2 rewards)."""
    err = np.abs(err)
    return float((np.mean(err <= 1) + np.mean(err <= 2)) / 2)


def stage_train(args):
    fi, split, bg, names = fold_meta(args.fold)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    wins = [tuple(w) for w in json.loads((EXP / "windows" / f"fold{fi}.json").read_text())["windows"]]
    train_p = sorted(split["train"])
    inner = set(random.Random(1234).sample(train_p, max(1, round(args.inner_frac * len(train_p)))))
    nsec = {p: n_seconds(p) for p in split["train"] + split["val"]}
    t0 = time.time()
    bank = FeatureBank(args.arm, fi, split["train"] + split["val"])
    xtr, ctr, ytr, _ = build_train_set(bank, [w for w in wins if w[0] not in inner], nsec, args.sigma)
    xin, cin, yin, info_in = build_train_set(bank, [w for w in wins if w[0] in inner], nsec, args.sigma)
    mu = xtr.reshape(-1, xtr.shape[-1]).astype(np.float32).mean(0)
    sd = xtr.reshape(-1, xtr.shape[-1]).astype(np.float32).std(0) + 1e-6
    norm = lambda x: ((x.astype(np.float32) - mu) / sd).astype(np.float32)
    xtr, xin = norm(xtr), norm(xin)
    model = Refiner(xtr.shape[-1], len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Xtr, Ctr, Ytr = (torch.from_numpy(a).to(device) for a in (xtr, ctr, ytr))
    t_in = np.array([i[1] for i in info_in])
    c_in = np.array([i[2] for i in info_in])
    pos_in = np.stack([i[3] for i in info_in])
    log, best = [], None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for s in range(0, len(perm), args.batch_size):
            b = perm[s : s + args.batch_size]
            loss = -(Ytr[b] * model(Xtr[b], Ctr[b])).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        prob = predict(model, xin, cin, device)
        in_loss = float(-(yin * np.log(prob + 1e-12)).sum(1).mean())
        score = hit_score(refine_times(prob, pos_in, c_in, 0.0) - t_in)
        log.append({"epoch": ep, "train_loss": tot / len(Xtr), "inner_loss": in_loss, "inner_hit": score})
        if best is None or in_loss < best[0]:
            best = (in_loss, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    prob = predict(model, xin, cin, device)
    conf = prob.max(1)
    cands = [0.0] + np.quantile(conf, np.linspace(0.05, 0.95, 19)).tolist()
    scores = [hit_score(refine_times(prob, pos_in, c_in, th) - t_in) for th in cands]
    thr = float(cands[int(np.argmax(scores))])
    inner_stats = {
        "inner_hit_before": hit_score(c_in - t_in),
        "inner_hit_after": float(max(scores)),
        "inner_share_le1_before": float(np.mean(np.abs(c_in - t_in) <= 1)),
        "inner_share_le1_after": float(np.mean(np.abs(refine_times(prob, pos_in, c_in, thr) - t_in) <= 1)),
        "inner_mae_after": float(np.mean(np.abs(refine_times(prob, pos_in, c_in, thr) - t_in))),
    }

    # ---- refine F_base detections of the val videos -------------------------------
    dets, lengths, meta = fbase_detections(args.seed, fi, bg)
    out_dets, r0_dets, moved, n_ref = {}, {}, [], 0
    for p, dd in dets.items():
        idx = [i for i, (c, t, s) in enumerate(dd) if s >= args.min_score]
        new = [list(map(float, d)) for d in dd]
        if idx:
            pos = np.stack([window_positions(float(dd[i][1]), bank.per_second, nsec[p]) for i in idx])
            x = norm(np.stack([bank.window(p, ps) for ps in pos]))
            c = np.array([dd[i][0] for i in idx])
            t_hat = np.array([float(dd[i][1]) for i in idx])
            pr = predict(model, x, c, device)
            t_new = refine_times(pr, pos, t_hat, thr)
            for k, i in enumerate(idx):
                new[i][1] = float(t_new[k])
            moved.extend((t_new - t_hat).tolist())
            n_ref += len(idx)
        out_dets[p] = [[int(c), t, s] for c, t, s in new]
        r0_dets[p] = [[int(c), float(t), float(s)] for c, t, s in dd]
    for arm, dd in ((args.arm, out_dets), ("R0", r0_dets)):
        d = EXP / arm / f"seed{args.seed}" / f"fold{fi}" / "val"
        d.mkdir(parents=True, exist_ok=True)
        (d / "detections.json").write_text(json.dumps({"patients": meta["patients"], "lengths": lengths,
                                                         "bg": list(bg), "n_classes": len(names), "detections": dd}))
    run_dir = EXP / args.arm / f"seed{args.seed}" / f"fold{fi}"
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "thr": thr, "args": vars(args)}, run_dir / "refiner.pt")
    moved = np.asarray(moved)
    summary = {
        "arm": args.arm, "seed": args.seed, "fold": args.fold, "per_second": bank.per_second,
        "train_windows": len(xtr), "inner_windows": len(xin), "inner_videos": sorted(inner),
        "best_epoch": best[1], "threshold": thr, **inner_stats,
        "val_refined": n_ref, "val_moved_share": float(np.mean(moved != 0)) if n_ref else 0.0,
        "val_mean_abs_move": float(np.mean(np.abs(moved))) if n_ref else 0.0,
        "time_s": time.time() - t0, "log": log,
    }
    (run_dir / "train_log.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["positions", "train"])
    ap.add_argument("--fold", default="0/5")
    ap.add_argument("--arm", default="R1", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0, help="refiner seed = F_base seed it refines")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="F_base seeds (positions stage)")
    ap.add_argument("--jitters", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--inner-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    stage_positions(args) if args.stage == "positions" else stage_train(args)


if __name__ == "__main__":
    main()
