#!/usr/bin/env python3
"""
EXP-22: L-A3 (BiMamba2 + MS-TCN, tools/train_long_mamba.py) on the 4 Hz EXP-19 arm S encoder latents, one model
for all 8 target events.

Same architecture, optimiser, schedule, augmentation and loss as EXP-21 (``train_long_mamba.py``: 100 epochs, AdamW
5e-4 cosine, crop 16384 rows, feature dropout 0.5, targets dilated +-1 s, weighted CE with inverse sqrt frequency,
phase CE, 0.15 T-MSE); the differences are (1) the event head has Idle + the 4 short + the 4 long targets (9 outputs,
the column order of ``tapis.datasets.focus8.EVENT_CLASSES``) and (2) the selection score on the inner val:
mean of the short-group and long-group spotting mAP, each averaged over +-1/2/5 s (NMS decoder).

Writes ``<out>/{inner_val,held_out}/<RARPxx>.npz`` (``second``, ``probs_steps`` [T, 9], ``probs_phases`` [T, 15]) +
``meta.json`` (the format of tools/eval_focus8.py), ``best.pt``, ``train_log.json``, ``.done``. Resumable.

    python tools/train_long_s.py --feat-dir <S fold features> --splits results/exp22/fold0/splits.json \
        --out results/exp22/fold0/a3_s0 --seed 0 [--causal]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "tools", REPO / "detectron2"):
    sys.path.insert(0, str(p))

import eval_focus8 as ev  # noqa: E402
import train_long_mamba as T  # noqa: E402  (A3, losses, predict: unchanged EXP-21 code)
from tapis.datasets import focus8 as F8  # noqa: E402
from tapis.evaluate import spotting_eval as se  # noqa: E402

NAMES = F8.EVENT_CLASSES  # Idle + 4 short + 4 long
GROUPS = {"short": [i for i, n in enumerate(NAMES) if n in F8.SHORT], "long": [i for i, n in enumerate(NAMES) if n in F8.LONG]}


def video_data(p, feat_dir, label_dir, dilate_rows):
    z = np.load(Path(feat_dir) / f"{p}.npz")
    x = z["features"]
    lab = F8.VideoLabels(Path(label_dir) / p / f"{p}_all_labels.csv", fps=4)
    assert len(x) == lab.n, (p, len(x), lab.n)
    y = np.zeros(lab.n, np.int64)
    for t, k in lab.events:
        if k > 0:  # k = index of the target in F8.TARGETS + 1 = column of NAMES
            r = int(round(t * 4))
            y[max(0, r - dilate_rows): r + dilate_rows + 1] = k
    gts = [(k, float(t)) for t, k in lab.events if k > 0 and not lab.ignore[min(lab.n - 1, int(round(t * 4)))]]
    return {"x": x, "y": y, "ph": lab.phase, "m": ~lab.ignore, "second": z["second"], "gts": gts}


def group_maps(preds, data, videos):
    """{group: {delta: mAP}} of NMS detections (same matching and AP as tools/eval_focus8.py)."""
    dets = {v: [(c, float(data[v]["second"][t]), s) for c, t, s in
                se.decode_events(preds[v][0], se.window_frames(5.0, 4), 0.0, background_classes=(0,))] for v in videos}
    gts = {v: data[v]["gts"] for v in videos}
    classes = list(range(1, len(NAMES)))
    m, n = ev.matches(dets, gts, classes)
    ap = ev.ap_table(m, n, sorted(videos), classes)
    return ev.group_map(ap, GROUPS)


def select_score(gm):
    """Mean of the two group mAPs, each averaged over +-1/2/5 s (nan groups ignored).

    >>> round(select_score({"short": {1: 0.2, 2: 0.4, 5: 0.6}, "long": {1: 0.0, 2: 0.2, 5: 0.4}}), 6)
    0.3
    """
    return float(np.nanmean([np.nanmean(list(v.values())) for v in gm.values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--label-dir", default=T.LAB4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--crop", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--feat-dropout", type=float, default=0.5)
    ap.add_argument("--dilate-s", type=float, default=1.0)
    ap.add_argument("--tmse", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--mamba-layers", type=int, default=4)
    ap.add_argument("--d-state", type=int, default=32)
    ap.add_argument("--headdim", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="2 train videos, 1 inner-val video, 2 epochs")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads(Path(args.splits).read_text())
    train, inner, held = sp["train"], sp["inner_val"], sp["held_out"]
    if args.smoke:
        train, inner, held, args.epochs, args.eval_every = train[:2], inner[:1], held[:1], 2, 1
    dev = torch.device("cuda", 0)
    dil = int(round(args.dilate_s * 4))
    t0 = time.time()
    data = {p: video_data(p, args.feat_dir, args.label_dir, dil) for p in train + inner + held}
    xs = np.concatenate([data[p]["x"] for p in train]).astype(np.float32)
    mu, sd = xs.mean(0), xs.std(0) + 1e-6
    ys = np.concatenate([data[p]["y"][data[p]["m"]] for p in train])
    freq = np.bincount(ys, minlength=len(NAMES)) / len(ys)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    w_ev = torch.tensor(w / (freq * w).sum(), dtype=torch.float32, device=dev)
    del xs, ys
    on_dev = {p: (torch.from_numpy(((d["x"].astype(np.float32) - mu) / sd).astype(np.float16)).T[None].to(dev),
                  torch.from_numpy(d["y"])[None].to(dev), torch.from_numpy(d["ph"])[None].to(dev),
                  torch.from_numpy(d["m"].astype(np.float32))[None].to(dev)) for p, d in data.items()}
    print(f"data {len(train)}/{len(inner)}/{len(held)} videos in {time.time() - t0:.0f}s, event weights "
          f"{[round(float(v), 2) for v in w_ev]}", flush=True)

    n_ev, n_ph, dim = len(NAMES), len(F8.PHASE_IDS), len(mu)
    model = T.A3(dim, n_ev, n_ph, args.stages, args.causal,
                 dict(d_model=args.d_model, layers=args.mamba_layers, d_state=args.d_state, headdim=args.headdim)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start, best, log = 0, -1.0, []
    last = out / "last.pt"
    if last.exists() and not args.smoke:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, log = ck["epoch"] + 1, ck["best"], ck["log"]
        random.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at epoch {start}, best {best:.4f}", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    cfg = dict(vars(args), n_params=n_par, event_classes=NAMES, mu_sd_from="train")

    for epoch in range(start, args.epochs):
        model.train()
        te, tot = time.time(), 0.0
        order = list(train)
        random.shuffle(order)
        for p in order:
            x, y, ph, m = on_dev[p]
            n = x.shape[-1]
            s = random.randint(0, max(0, n - args.crop))
            sl = slice(s, s + args.crop)
            xc = x[..., sl].float()
            xc = F.dropout1d(xc, args.feat_dropout, training=True) if args.feat_dropout else xc
            loss = T.losses(model(xc), y[:, sl], ph[:, sl], m[:, sl], n_ev, w_ev, args.tmse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()
        rec = {"epoch": epoch, "train_loss": tot / len(order), "train_s": round(time.time() - te, 1)}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            preds = {p: T.predict(model, on_dev[p][0].float(), n_ev) for p in inner}
            gm = group_maps(preds, data, inner)
            acc = float(np.mean(np.concatenate([preds[p][1].argmax(1) == data[p]["ph"] for p in inner])))
            metric = select_score(gm)
            rec.update(val_group_map={g: {str(d): v for d, v in x.items()} for g, x in gm.items()},
                       val_metric=metric, val_phase_acc=acc)
            if metric > best:
                best = metric
                torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "cfg": cfg, "epoch": epoch}, out / "best.pt")
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if not args.smoke:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "best": best, "log": log, "py_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state()}, last)
    (out / "train_log.json").write_text(json.dumps({"cfg": cfg, "log": log}, indent=1))

    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    for name, vids in (("inner_val", inner), ("held_out", held)):
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"fps": 4, "event_classes": NAMES, "best_epoch": ck["epoch"],
                                                 "feat_dir": str(Path(args.feat_dir).resolve()), "patients": vids,
                                                 "cfg": cfg}, indent=1, default=str))
        for p in vids:
            e, ph = T.predict(model, on_dev[p][0].float(), n_ev)
            np.savez_compressed(d / f"{p}.npz", second=data[p]["second"], probs_steps=e.astype(np.float32),
                                probs_phases=ph.astype(np.float32))
    if not args.smoke:
        (out / ".done").touch()
    print(f"done: best epoch {ck['epoch']} val metric {best:.4f}, {n_par} params", flush=True)


if __name__ == "__main__":
    main()
