#!/usr/bin/env python3
"""
EXP-21 phase 3, EXPLORATORY (not pre-registered): temporal MS-TCN for the SHORT events on the arm M 4 Hz latents.

Same recipe as the reference EXP-19 S MS-TCN dilate3 (tools/train_temporal.py: dilated targets of +-3 s, focal BCE
per event, phase head with CE + 0.15 T-MSE, 2 stages, 64 channels, 10 layers, dropout 0.5, feature dropout 0.5,
Adam 5e-4 + cosine, 100 epochs, crop 16384, the LAST epoch is used, no checkpoint selection), but on the focus8
setup: features ``<feat-dir>/<RARPxx>.npz`` (key ``features``, 1536-d), splits.json (train / inner_val / held_out),
revised 4 Hz labels, event head = background + the 8 target events (non-target events are not classes), rows with
``ignore=1`` masked. This gives the fair comparison with the reference MS-TCN (short mAP@2 0.738), whose
temporal model sees the sequence, against the per-clip encoder heads of arm M.

Predictions for inner_val and held_out are written as ``<out>/<set>/<RARPxx>.npz`` (``second``, ``probs_steps``
[T, 9], ``probs_phases`` [T, 15]) + ``meta.json``: the format of tools/eval_focus8.py.

    python tools/train_short_mstcn.py --feat-dir results/M/fold0/untrimmed --splits results/M/fold0/splits.json \
        --out results/p3/short_mstcn/s0 --seed 0
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "tools", REPO / "detectron2"):
    sys.path.insert(0, str(p))

from tapis.datasets import focus8 as F8  # noqa: E402
from train_temporal import TemporalModel, event_targets, losses, predict  # noqa: E402

LAB4 = "/home/gchie/workspace/data/orsi_labels_revised/v2026-11-19_4hz"
BG, FG = (0,), list(range(1, 1 + len(F8.TARGETS)))


def video_data(p, feat_dir, label_dir, k_rows, fractional=True):
    z = np.load(Path(feat_dir) / f"{p}.npz")
    x = z["features"].astype(np.float32)
    lab = F8.VideoLabels(Path(label_dir) / p / f"{p}_all_labels.csv", fps=4)
    assert len(x) == lab.n, (p, len(x), lab.n)
    ev = [(k, float(t) * 4) for t, k in lab.events if k > 0 and not lab.ignore[min(lab.n - 1, int(round(t * 4)))]]
    y, m = event_targets(lab.n, ev, len(F8.TARGETS) + 1, "dilate", k_rows)
    m[lab.ignore] = 0.0
    return {"x": x, "phase": lab.phase.astype(np.int64), "y": y, "m": m, "second": z["second"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--label-dir", default=LAB4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--feat-dropout", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=16384)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--smooth-weight", type=float, default=0.15)
    ap.add_argument("--k-sec", type=float, default=3.0)
    ap.add_argument("--causal", action="store_true", help="left-only padding in every dilated conv (online model, EXP-25)")
    ap.add_argument("--lookahead", default="", help="comma-separated look-aheads in seconds (EXP-25a): one model per value in <out>/F<value>/, features loaded once; implies --causal use")
    ap.add_argument("--smoke", action="store_true", help="2 train videos, 1 inner-val, 1 held-out, 2 epochs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads(Path(args.splits).read_text())
    train, inner, held = sp["train"], sp["inner_val"], sp["held_out"]
    if args.smoke:
        train, inner, held, args.epochs = train[:2], inner[:1], held[:1], 2
    dev = torch.device("cuda", 0)
    k_rows = int(round(args.k_sec * 4))
    t0 = time.time()
    data = {p: video_data(p, args.feat_dir, args.label_dir, k_rows) for p in train + inner + held}
    xs = np.concatenate([data[p]["x"] for p in train])
    mu, sd = xs.mean(0), xs.std(0) + 1e-6
    del xs
    # normalised features stay in CPU RAM (a crop is copied to the GPU per step): the GPU needs about 1 GB instead of 14 GB
    cpu = {}
    for p in list(data):
        d = data[p]
        cpu[p] = (torch.from_numpy((d["x"] - mu) / sd).T.contiguous()[None], torch.from_numpy(d["phase"])[None],
                  torch.from_numpy(d["y"]).T.contiguous()[None], torch.from_numpy(d["m"]).T.contiguous()[None])
        del d["x"], d["y"], d["m"], d["phase"]
    print(f"data {len(train)}/{len(inner)}/{len(held)} videos in {time.time() - t0:.0f}s", flush=True)
    looks = [float(x) for x in args.lookahead.split(",")] if args.lookahead else [None]
    for look in looks:
        train_one(args, cpu, data, train, inner, held, mu, dev, out / (f"F{look:g}" if look is not None else ""),
                  int(round(4 * look)) if look is not None else 0)
    print("done", flush=True)


def shifted(t, S, pad_value):
    """Labels of row t - S at row t (``S`` rows of ``pad_value`` first): [.., T] -> [.., T]."""
    if S == 0:
        return t
    pad = torch.full_like(t[..., :S], pad_value)
    return torch.cat([pad, t[..., :-S]], -1)


def train_one(args, cpu, data, train, inner, held, mu, dev, out, S):
    """One model with a look-ahead of S rows (S = 0: the EXP-25 causal model), seeded with ``args.seed`` (independent of other S)."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    # model output at row t is trained against the labels of row t - S; the first S rows have no label (mask 0, phase ignored)
    tr = {p: (x, shifted(yp, S, int(yp[0, 0])), shifted(ye, S, 0.0), shifted(me, S, 0.0)) for p, (x, yp, ye, me) in cpu.items()}
    model = TemporalModel(len(mu), len(F8.PHASE_IDS), len(F8.TARGETS) + 1, args.channels, args.layers, args.stages,
                          args.causal, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start, log = 0, []
    last = out / "last.pt"
    if last.exists() and not args.smoke:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, log = ck["epoch"] + 1, ck["log"]
        random.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at epoch {start}", flush=True)
    for epoch in range(start, args.epochs):
        model.train()
        te, tot = time.time(), 0.0
        order = list(train)
        random.shuffle(order)
        for p in order:
            x, yp, ye, me = tr[p]
            s = random.randint(0, max(0, x.shape[-1] - args.crop))
            sl = slice(s, s + args.crop)
            xc = x[..., sl].to(dev, non_blocking=True)
            xc = Fn.dropout1d(xc, args.feat_dropout, training=True) if args.feat_dropout else xc
            loss = losses(model(xc), yp[:, sl].to(dev), ye[..., sl].to(dev), me[..., sl].to(dev), FG, args.gamma, args.smooth_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        rec = {"epoch": epoch, "train_loss": tot / len(order), "train_s": round(time.time() - te, 1)}
        log.append(rec)
        print(json.dumps(dict(rec, lookahead_rows=S)), flush=True)
        if not args.smoke:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "log": log, "py_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state()}, last)
    cfg = dict(vars(args), event_classes=F8.EVENT_CLASSES, selection="last epoch (reference recipe)", lookahead_rows=S)
    (out / "train_log.json").write_text(json.dumps({"cfg": cfg, "log": log}, indent=1))
    model.eval()
    for name, vids in (("inner_val", inner), ("held_out", held)):
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"fps": 4, "event_classes": F8.EVENT_CLASSES, "patients": vids,
                                                 "feat_dir": str(Path(args.feat_dir).resolve()), "cfg": cfg},
                                                indent=1, default=str))
        for p in vids:
            ev, ph = predict(model, cpu[p][0].to(dev), BG, FG)
            if S:   # read the output S rows later (the last S rows repeat the final output)
                idx = np.minimum(np.arange(len(ev)) + S, len(ev) - 1)
                ev, ph = ev[idx], ph[idx]
            np.savez_compressed(d / f"{p}.npz", second=data[p]["second"], probs_steps=ev.astype(np.float32),
                                probs_phases=ph.astype(np.float32))
    if not args.smoke:
        (out / ".done").touch()


if __name__ == "__main__":
    main()
