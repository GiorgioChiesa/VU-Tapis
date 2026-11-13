#!/usr/bin/env python3
"""
Experiment 7: a temporal model dedicated to the Orsi phases (no event head).

Same features, folds (`train_hybrid.cv_split`), seeds, optimiser, schedule, crop and input dropout as
F_base (`train_hybrid.py --context G2 --target dilate --k 3`, EXP2/F_base); only the phase head is trained.

Arms (``--arm`` sets the defaults, every knob can be overridden):
  A1  MS-TCN, 2 stages, CE + 0.15 T-MSE (the phase loss of F_base): phase-only vs shared head
  A2  MS-TCN, 4 stages, CE weighted by 1/sqrt(train-fold frequency) + 0.15 T-MSE
  A3  bidirectional Mamba2 encoder (`tapis/models/mamba2_minimal.py`, pure PyTorch) as stage 1
      + 3 MS-TCN refinement stages, A2 losses
Refinement stages read the detached softmax of the previous stage (as `train_temporal.TemporalModel`).

Model selection never sees the test fold: pass 1 trains on the inner-train videos (training videos of the
fold minus an inner validation split of ``--inner-frac``, chosen by Random(4321 + fold)) and evaluates the
inner validation videos every ``--select-every`` epochs with the per-video seg F1@50 of the argmax phases;
the best epoch E* (earliest on ties) is kept. Pass 2 refits on all training videos of the fold with the
same seed and schedule and keeps the weights at epoch E* (selected) and at the last epoch.

Outputs, in --out (= <arm>/seed<s>/fold<i>):
  phase_posteriors/<video>.npz + classes.json    test-fold posteriors of the selected model (shared format,
                                                 see tools/phase_metrics.py)
  phase_posteriors_last/                          same, last epoch (F_base protocol: no selection)
  model.pt, train_log.json (inner-validation curve, E*), metrics.json (aggregate test-fold metrics)

Example:
    python tools/train_phase.py --arm A2 --cv-fold 0/5 --seed 0 --device cpu --threads 2 \
        --feat-dir outputs/orsi/LONG/cv_ft/fold0/untrimmed/checkpoint_best_steps/all \
        --out outputs/orsi/EXP7/A2/seed0/fold0
"""

import argparse
import copy
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

import phase_metrics as pm
from train_hybrid import load_orsi
from train_temporal import DilatedResidual
from tapis.models.mamba2_minimal import BiMamba2Encoder

ARMS = {
    "A1": dict(encoder="mstcn", stages=2, class_weight="none"),
    "A2": dict(encoder="mstcn", stages=4, class_weight="invsqrt"),
    "A3": dict(encoder="bimamba2", stages=4, class_weight="invsqrt"),
}


# ---- model -------------------------------------------------------------------------
class PhaseStage(nn.Module):
    """MS-TCN stage with a phase head only (`train_temporal.Stage` without the event head)."""

    def __init__(self, dim_in, channels, layers, n_classes, dropout):
        super().__init__()
        self.inp = nn.Conv1d(dim_in, channels, 1)
        self.layers = nn.ModuleList(DilatedResidual(2**i, channels, False, dropout) for i in range(layers))
        self.out = nn.Conv1d(channels, n_classes, 1)

    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


class PhaseModel(nn.Module):
    """
    ``x [B, D, T] -> [phase logits [B, C, T]]`` per stage (the last one is the prediction).

    >>> m = PhaseModel(20, 5, "mstcn", stages=3, channels=8, layers=3)
    >>> [o.shape[1:] for o in m(torch.randn(1, 20, 50))]
    [torch.Size([5, 50]), torch.Size([5, 50]), torch.Size([5, 50])]
    >>> m = PhaseModel(20, 5, "bimamba2", stages=2, channels=8, layers=3, mamba=dict(d_model=8, layers=1, d_state=4, headdim=4))
    >>> [o.shape[1:] for o in m(torch.randn(1, 20, 50))]
    [torch.Size([5, 50]), torch.Size([5, 50])]
    """

    def __init__(self, dim_in, n_classes, encoder, stages, channels=64, layers=10, dropout=0.5, mamba=None):
        super().__init__()
        first = (PhaseStage(dim_in, channels, layers, n_classes, dropout) if encoder == "mstcn"
                 else BiMamba2Encoder(dim_in, n_classes, **(mamba or {})))
        self.stages = nn.ModuleList([first] + [PhaseStage(n_classes, channels, layers, n_classes, dropout)
                                               for _ in range(stages - 1)])

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            outs.append(stage(x if i == 0 else F.softmax(outs[-1], 1).detach()))
        return outs


def phase_loss(outs, y, weight, smooth_w):
    """Sum over stages of (weighted) CE + smooth_w * T-MSE (the phase terms of `train_temporal.losses`)."""
    total = 0.0
    for p in outs:
        logp = F.log_softmax(p, 1)
        tmse = torch.clamp((logp[:, :, 1:] - logp.detach()[:, :, :-1]) ** 2, max=16).mean()
        total = total + F.cross_entropy(p, y, weight=weight) + smooth_w * tmse
    return total


def class_weights(labels, n_classes, kind):
    """
    Inverse sqrt frequency, scaled so that the frame-weighted mean is 1 (same loss scale as plain CE).

    >>> class_weights(np.array([0, 0, 0, 1]), 3, "invsqrt").round(3).tolist()
    [0.845, 1.464, 0.0]
    """
    if kind == "none":
        return None
    freq = np.bincount(labels, minlength=n_classes) / len(labels)
    w = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1e-12)), 0.0)
    return w / (freq * w).sum()


def inner_split(train, fold, frac):
    """
    Inner validation videos of a fold: ``round(frac * len(train))`` drawn by Random(4321 + fold).

    >>> a, b = inner_split([f"v{i}" for i in range(10)], 2, 0.2)
    >>> len(a), len(b), sorted(a + b) == sorted(f"v{i}" for i in range(10))
    (8, 2, True)
    """
    vids = sorted(train)
    random.Random(4321 + fold).shuffle(vids)
    held = set(vids[: int(round(frac * len(vids)))])
    return [v for v in train if v not in held], [v for v in train if v in held]


# ---- training ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv-fold", required=True, help="i/n video-level fold (train_hybrid.cv_split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    ap.add_argument("--encoder", choices=["mstcn", "bimamba2"])
    ap.add_argument("--stages", type=int)
    ap.add_argument("--class-weight", choices=["none", "invsqrt"])
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--mamba-dim", type=int, default=64)
    ap.add_argument("--mamba-layers", type=int, default=4)
    ap.add_argument("--mamba-state", type=int, default=32)
    ap.add_argument("--mamba-dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--feat-dropout", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=4096)
    ap.add_argument("--smooth-weight", type=float, default=0.15)
    ap.add_argument("--inner-frac", type=float, default=0.2)
    ap.add_argument("--select-every", type=int, default=5)
    ap.add_argument("--max-epochs-debug", type=int, help="stop both passes after this many epochs (smoke test)")
    args = ap.parse_args(argv)
    for k, v in ARMS[args.arm].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def train_pass(args, data, train, n_classes, device, select_on=None):
    """
    Train on ``train`` for args.epochs; returns ``(model, posteriors fn, log, snapshots)``. With ``select_on``
    (inner validation videos) the log has the inner metrics every ``select_every`` epochs; without it, the
    state dicts at those epochs (and the last) are kept in ``snapshots``.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x_all = np.concatenate([data[p]["x"] for p in train])
    mu, sd = x_all.mean(0), x_all.std(0) + 1e-6
    del x_all
    w = class_weights(np.concatenate([data[p]["phase"] for p in train]), n_classes, args.class_weight)
    weight = None if w is None else torch.as_tensor(w, dtype=torch.float32, device=device)
    mamba = dict(d_model=args.mamba_dim, layers=args.mamba_layers, d_state=args.mamba_state, dropout=args.mamba_dropout)
    model = PhaseModel(len(mu), n_classes, args.encoder, args.stages, args.channels, args.layers, args.dropout,
                       mamba).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def tensor(p, start=0, stop=None):
        d, sl = data[p], slice(start, stop)
        return (torch.from_numpy((d["x"][sl] - mu) / sd).T[None].float().to(device),
                torch.from_numpy(d["phase"][sl])[None].long().to(device))

    @torch.no_grad()
    def posteriors(videos):
        model.eval()
        return {p: F.softmax(model(tensor(p)[0])[-1], 1)[0].T.cpu().numpy() for p in videos}

    log, snapshots, t0 = [], {}, time.time()
    n_epochs = min(args.epochs, args.max_epochs_debug or args.epochs)
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = list(train)
        random.shuffle(order)
        ep_loss = 0.0
        for p in order:
            n = len(data[p]["x"])
            start = random.randint(0, max(0, n - args.crop))
            x, y = tensor(p, start, start + args.crop)
            x = F.dropout1d(x, args.feat_dropout, training=True) if args.feat_dropout else x
            loss = phase_loss(model(x), y, weight, args.smooth_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        row = {"epoch": epoch, "train/loss": ep_loss / len(train), "time_s": time.time() - t0}
        if epoch % args.select_every == 0 or epoch == n_epochs:
            if select_on is not None:
                post = posteriors(select_on)
                m = pm.run_metrics({p: v.argmax(1) for p, v in post.items()},
                                   {p: data[p]["phase"] for p in select_on}, n_classes)
                row.update({f"inner/{k}": m[k] for k in pm.PER_VIDEO})
            else:
                snapshots[epoch] = copy.deepcopy(model.state_dict())
            log.append(row)
            print(f"epoch {epoch}: loss {row['train/loss']:.4f}"
                  + (f" inner F1@50 {row['inner/F1@50']:.1f} edit {row['inner/edit']:.1f}" if select_on else "")
                  + f" ({row['time_s']:.0f}s)", flush=True)
    return model, posteriors, log, snapshots


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fold = int(args.cv_fold.split("/")[0])
    ns = argparse.Namespace(feat_dir=args.feat_dir, frame_dir=args.frame_dir, cv_fold=args.cv_fold, target="point",
                            k=3, sigma=2.0, streams="dynamic")
    data, split, info = load_orsi(ns, out)
    classes = info["ph_names"]
    n_classes = len(classes)
    for d in data.values():  # events are not used
        d.pop("y", None), d.pop("m", None)
    train, test = split["train"], split["val"]  # cv_split names the held-out fold "val"
    inner_train, inner_val = inner_split(train, fold, args.inner_frac)
    print(f"{args.arm} fold {args.cv_fold} seed {args.seed}: {len(inner_train)} inner-train, {len(inner_val)} "
          f"inner-val, {len(train)} train, {len(test)} test videos; {n_classes} phases", flush=True)

    t0 = time.time()
    print("pass 1 (selection on the inner validation split)", flush=True)
    _, _, log1, _ = train_pass(args, data, inner_train, n_classes, device, select_on=inner_val)
    best = max(log1, key=lambda r: (r["inner/F1@50"], -r["epoch"]))
    e_star = best["epoch"]
    print(f"E* = {e_star} (inner F1@50 {best['inner/F1@50']:.1f})", flush=True)

    print("pass 2 (refit on all training videos)", flush=True)
    model, posteriors, log2, snaps = train_pass(args, data, train, n_classes, device)
    last = max(snaps)
    gts = {p: data[p]["phase"] for p in test}
    metrics = {}
    for tag, ep in (("selected", e_star), ("last", last)):
        model.load_state_dict(snaps[ep])
        post = posteriors(test)
        sub = "phase_posteriors" if tag == "selected" else "phase_posteriors_last"
        pm.write_posteriors(out, post, gts, fold, classes, sub)
        metrics[tag] = {"epoch": ep, **pm.run_metrics({p: v.argmax(1) for p, v in post.items()}, gts, n_classes)}
    model.load_state_dict(snaps[e_star])
    n_params = sum(p.numel() for p in model.parameters())
    torch.save({"model": model.state_dict(), "epoch": e_star, "args": vars(args), "classes": classes}, out / "model.pt")
    (out / "classes.json").write_text(json.dumps(classes))
    total = time.time() - t0
    (out / "train_log.json").write_text(json.dumps({
        "args": vars(args), "e_star": e_star, "inner_train": inner_train, "inner_val": inner_val,
        "pass1": log1, "pass2": log2, "n_params": n_params, "time_s": total}, indent=1))
    (out / "metrics.json").write_text(json.dumps({"arm": args.arm, "fold": fold, "seed": args.seed,
                                                  "n_params": n_params, "time_s": total, **metrics}, indent=1))
    s = metrics["selected"]
    print(f"done {out}: E*={e_star} test F1@50 {s['F1@50']:.1f} edit {s['edit']:.1f} macroF1 {s['macro_F1']:.3f} "
          f"acc {s['acc']:.3f} ratio {s['seg_ratio']:.2f} ({total:.0f}s, {n_params / 1e6:.3f} M params)", flush=True)


if __name__ == "__main__":
    main()
