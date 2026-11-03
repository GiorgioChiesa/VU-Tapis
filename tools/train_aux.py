#!/usr/bin/env python3
"""
Experiment 2: auxiliary predictors trained on the external RARP datasets, applied to Orsi.

One MS-TCN per source dataset, on the same frozen GraSP-LONG per-second features used for
Orsi, with one output group per task:

  grasp  grasp_phase (11, softmax), grasp_step (21, softmax),
         grasp_action (14, multi-label, keyframes only), grasp_instrument (7, multi-label, keyframes only)
  sar    sar_gesture (8, softmax; DVC suturing segments only)
  esad   esad_action (21, multi-label; dense)

Labels come from `tools/prepare_external.py` (`data/external_1fps/<ds>/<video>/labels.npz`),
features from `infer_untrimmed.py --dataset <ds> --features` (`outputs/external/<ds>/grasp_LONG`).

Modes:
  --mode eval   train on the official train (+val) split, report quality on the test split
                (accuracy and frame mAP for softmax tasks, keyframe/frame mAP for multi-label)
  --mode final  train on every video of the dataset and write the per-second posteriors of all
                50 Orsi videos to <out>/<patient>.npz (one array per task), used by
                `train_hybrid.py --aux-mode stack`. The aux model never sees Orsi.

Example:
    python tools/train_aux.py --dataset grasp --mode eval  --out outputs/orsi/AUX/eval/grasp --gpu 2
    python tools/train_aux.py --dataset grasp --mode final --out outputs/orsi/AUX/final/grasp --gpu 2
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_temporal import DilatedResidual

LABELS = Path("/home/gchie/workspace/data/external_1fps")
FEATS = Path(__file__).resolve().parents[1] / "outputs" / "external"
ORSI_FEATS = Path(__file__).resolve().parents[1] / "outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all"

# task -> (dataset, label key, kind, mask key or None)
TASKS = {
    "grasp_phase": ("grasp", "phases", "softmax", None),
    "grasp_step": ("grasp", "steps", "softmax", None),
    "grasp_action": ("grasp", "actions", "multilabel", "actions_mask"),
    "grasp_instrument": ("grasp", "instruments", "multilabel", "actions_mask"),
    "sar_gesture": ("sar_rarp50", "gestures", "softmax", None),
    "esad_action": ("esad", "actions", "multilabel", None),
}
DATASET_OF = {"grasp": "grasp", "sar": "sar_rarp50", "esad": "esad"}
# --aux-sources of train_hybrid.py -> tasks
SOURCES = {
    "grasp_phase": ["grasp_phase", "grasp_step"],
    "grasp_action": ["grasp_action", "grasp_instrument"],
    "sar": ["sar_gesture"],
    "esad": ["esad_action"],
}


def tasks_of(dataset):
    return [t for t, (d, *_) in TASKS.items() if d == DATASET_OF[dataset]]


def vocab_sizes(dataset):
    meta = json.loads((LABELS / DATASET_OF[dataset] / "meta.json").read_text())
    key = {t: TASKS[t][1] for t in tasks_of(dataset)}
    return {t: len(meta["vocab"][k]) for t, k in key.items()}, meta


def load_video(dataset, video, tasks):
    """Features ``[T, D]`` and per-task targets: int ``[T]`` (softmax) or float ``[T, n]`` + mask ``[T]``."""
    ds = DATASET_OF[dataset]
    lab = np.load(LABELS / ds / video / "labels.npz")
    npz = np.load(FEATS / ds / "grasp_LONG" / f"{video}.npz")
    if not np.array_equal(lab["second"], npz["second"]):
        raise RuntimeError(f"{video}: labels and features are not aligned")
    y = {}
    for t in tasks:
        _, key, kind, mask = TASKS[t]
        if kind == "softmax":
            y[t] = lab[key].astype(np.int64)
        else:
            m = lab[mask].astype(np.float32) if mask else np.ones(len(lab["second"]), np.float32)
            y[t] = (lab[key].astype(np.float32), m)
    return npz["features"].astype(np.float32), y


class AuxTCN(nn.Module):
    """MS-TCN with one output group per task; refinement stages read softmax/sigmoid of the previous stage."""

    def __init__(self, dim_in, groups, channels=64, layers=10, stages=2, dropout=0.5):
        super().__init__()
        self.groups = groups  # [(task, kind, n)]
        n_out = sum(n for _, _, n in groups)
        self.stages = nn.ModuleList()
        for s in range(stages):
            self.stages.append(nn.ModuleDict({
                "inp": nn.Conv1d(dim_in if s == 0 else n_out, channels, 1),
                "layers": nn.Sequential(*[DilatedResidual(2**i, channels, False, dropout) for i in range(layers)]),
                "out": nn.Conv1d(channels, n_out, 1),
            }))

    def split(self, o):
        out, i = {}, 0
        for t, kind, n in self.groups:
            out[t] = o[:, i : i + n]
            i += n
        return out

    def activate(self, o):
        return torch.cat([F.softmax(g, 1) if kind == "softmax" else torch.sigmoid(g)
                          for (t, kind, _), g in zip(self.groups, self.split(o).values())], 1)

    def forward(self, x):  # [B, D, T] -> [ {task: [B, n, T]} per stage ]
        outs = []
        for s, st in enumerate(self.stages):
            if s:
                x = self.activate(outs[-1]).detach()
            outs.append(st["out"](st["layers"](st["inp"](x))))
        return [self.split(o) for o in outs]


def aux_loss(outs, targets, groups, smooth_w=0.15):
    total = 0.0
    for stage in outs:
        for t, kind, _ in groups:
            o = stage[t]
            if kind == "softmax":
                y = targets[t]
                total = total + F.cross_entropy(o, y)
                logp = F.log_softmax(o, 1)
                total = total + smooth_w * torch.clamp((logp[:, :, 1:] - logp.detach()[:, :, :-1]) ** 2, max=16).mean()
            else:
                y, m = targets[t]
                if m.sum() == 0:
                    continue
                bce = F.binary_cross_entropy_with_logits(o, y, reduction="none").mean(1)
                total = total + (bce * m).sum() / m.sum()
    return total


def average_precision(score, y):
    """AP of one binary problem (sklearn-free, ties broken by order).

    >>> round(average_precision(np.array([0.9, 0.8, 0.1]), np.array([1, 0, 1])), 3)
    0.833
    """
    order = np.argsort(-score, kind="stable")
    y = y[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def evaluate(preds, ys, groups):
    """Quality of each task on held-out videos (softmax: accuracy, frame mAP; multi-label: mAP over masked frames)."""
    res = {}
    for t, kind, n in groups:
        if kind == "softmax":
            y = np.concatenate([ys[v][t] for v in preds])
            p = np.concatenate([preds[v][t] for v in preds])
            aps = [average_precision(p[:, c], (y == c).astype(int)) for c in range(n)]
            res[t] = {"accuracy": float((p.argmax(1) == y).mean()), "frame_mAP": float(np.nanmean(aps)),
                      "per_class_ap": aps, "seconds": int(len(y))}
        else:
            m = np.concatenate([ys[v][t][1] for v in preds]) > 0
            y = np.concatenate([ys[v][t][0] for v in preds])[m]
            p = np.concatenate([preds[v][t] for v in preds])[m]
            aps = [average_precision(p[:, c], y[:, c]) for c in range(n)]
            res[t] = {"mAP": float(np.nanmean(aps)), "per_class_ap": aps, "frames": int(m.sum()),
                      "positives_per_class": y.sum(0).astype(int).tolist()}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_OF))
    ap.add_argument("--mode", required=True, choices=["eval", "final"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--crop", type=int, default=4096)
    ap.add_argument("--feat-dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--orsi-feat-dir", default=str(ORSI_FEATS))
    args = ap.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = tasks_of(args.dataset)
    sizes, meta = vocab_sizes(args.dataset)
    groups = [(t, TASKS[t][2], sizes[t]) for t in tasks]
    splits = meta["splits"]
    test = splits["test"]
    train = [v for s, vs in splits.items() if s != "test" for v in vs]
    if args.mode == "final":
        train = train + test
    data = {v: load_video(args.dataset, v, tasks) for v in set(train + test)}

    xs = np.concatenate([data[v][0] for v in train])
    mu, sd = xs.mean(0), xs.std(0) + 1e-6
    model = AuxTCN(xs.shape[1], groups).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    def to_x(x):
        return torch.from_numpy((x - mu) / sd).T[None].float().to(device)

    def to_y(y, sl):
        out_y = {}
        for t, (_, _, kind, _) in ((t, TASKS[t]) for t in tasks):
            if kind == "softmax":
                out_y[t] = torch.from_numpy(y[t][sl])[None].to(device)
            else:
                out_y[t] = (torch.from_numpy(y[t][0][sl]).T[None].to(device), torch.from_numpy(y[t][1][sl])[None].to(device))
        return out_y

    model.train()
    for it in range(args.iters):
        v = random.choice(train)
        x, y = data[v]
        start = random.randint(0, max(0, len(x) - args.crop))
        sl = slice(start, start + args.crop)
        xt = F.dropout1d(to_x(x[sl]), args.feat_dropout, training=True)
        loss = aux_loss(model(xt), to_y(y, sl), groups)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if (it + 1) % 500 == 0:
            print(f"iter {it + 1}: loss {loss.item():.4f}", flush=True)

    model.eval()

    @torch.no_grad()
    def posteriors(x):
        last = model(to_x(x))[-1]
        return {t: (F.softmax(last[t], 1) if kind == "softmax" else torch.sigmoid(last[t]))[0].T.cpu().numpy()
                for t, kind, _ in groups}

    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "groups": groups, "args": vars(args)}, out / "model.pt")
    report = {"dataset": args.dataset, "mode": args.mode, "train": train, "test": test, "groups": groups}
    if args.mode == "eval":
        preds = {v: posteriors(data[v][0]) for v in test}
        report["test_metrics"] = evaluate(preds, {v: data[v][1] for v in test}, groups)
        for t, r in report["test_metrics"].items():
            print(t, {k: v for k, v in r.items() if not isinstance(v, list)}, flush=True)
    else:
        orsi = Path(args.orsi_feat_dir)
        names = sorted(p.stem for p in orsi.glob("RARP*.npz"))
        for p in names:
            npz = np.load(orsi / f"{p}.npz")
            post = posteriors(npz["features"].astype(np.float32))
            np.savez_compressed(out / f"{p}.npz", second=npz["second"], **{t: a.astype(np.float16) for t, a in post.items()})
        report["orsi_patients"] = names
        print(f"wrote posteriors for {len(names)} Orsi videos", flush=True)
    vocab = {t: meta["vocab"][TASKS[t][1]] for t in tasks}
    (out / "aux_report.json").write_text(json.dumps(dict(report, vocab=vocab), indent=1))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
