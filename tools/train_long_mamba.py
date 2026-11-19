#!/usr/bin/env python3
"""
EXP-21 long-context models on frozen 4 Hz encoder latents (``<feat-dir>/<RARPxx>.npz``, key ``features``,
from tools/infer_untrimmed.py --fps 4 --features).

Heads: long events (background + the 4 long targets, softmax; targets dilated +-``--dilate-s`` around each
annotation) and phases (15, softmax). Loss per stage: weighted CE on events (inverse sqrt frequency, rows in
ignore windows masked) + CE on phases + ``--tmse`` * truncated MSE on both heads.

Architectures (``--arch``):
  * ``a3``: BiMamba2Encoder (tapis/models/mamba2_minimal.py, pure PyTorch) + MS-TCN refinement stages;
    ``--causal``: forward scan only + causal (left padded) MS-TCN convolutions;
  * ``sed``: SEDMamba (official code, ``--sed-repo``, Apache-2.0; needs mamba_ssm, run in the mamba venv),
    BMSS blocks unchanged, its single classifier replaced by our two heads; ``--causal``: the FCTF dilated
    convolutions are left padded (2 * dilation), the Mamba scan is already causal.

Training on ``splits.json`` ``train``; checkpoint selection on ``inner_val`` (long-group spotting mAP averaged
over +-1/2/5 s, NMS decoder); predictions for ``inner_val`` and ``held_out`` written as
``<out>/<set>/<RARPxx>.npz`` (``second``, ``probs_steps`` [T, 5], ``probs_phases`` [T, 15]) + ``meta.json``,
the format of tools/eval_focus8.py. Resumable (``last.pt`` every epoch). Writes aggregates only to logs.

    python tools/train_long_mamba.py --arch a3 --feat-dir results/M/fold0/untrimmed \
        --splits results/M/fold0/splits.json --out results/M/fold0/long/a3_s0 --seed 0
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "tools"):
    sys.path.insert(0, str(p))

from tapis.datasets import focus8 as F8  # noqa: E402
from tapis.evaluate import spotting_eval as se  # noqa: E402

EVENT_CLASSES = ["Idle"] + F8.LONG
LAB4 = "/home/gchie/workspace/data/orsi_labels_revised/v2026-11-19_4hz"


# ---------------------------------------------------------------------------------------------- models
class DilatedResidual(nn.Module):
    def __init__(self, dilation, channels, causal, dropout):
        super().__init__()
        self.pad = (2 * dilation, 0) if causal else (dilation, dilation)
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.out(F.relu(self.conv(F.pad(x, self.pad)))))


class TCNStage(nn.Module):
    """MS-TCN refinement stage: input = softmax of the previous stage (events + phases)."""

    def __init__(self, dim_in, n_ev, n_ph, channels=64, layers=10, causal=False, dropout=0.5):
        super().__init__()
        self.inp = nn.Conv1d(dim_in, channels, 1)
        self.layers = nn.ModuleList(DilatedResidual(2**i, channels, causal, dropout) for i in range(layers))
        self.out = nn.Conv1d(channels, n_ev + n_ph, 1)

    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


class A3(nn.Module):
    """BiMamba2Encoder (or forward-only) + ``stages - 1`` MS-TCN stages. ``x [B, D, T] -> [logits [B, E+P, T]]``."""

    def __init__(self, dim_in, n_ev, n_ph, stages=3, causal=False, mamba=None, channels=64, layers=10, dropout=0.5):
        super().__init__()
        from tapis.models.mamba2_minimal import BiMamba2Encoder

        self.n_ev = n_ev
        self.first = BiMamba2Encoder(dim_in, n_ev + n_ph, bidirectional=not causal, **(mamba or {}))
        self.refine = nn.ModuleList(TCNStage(n_ev + n_ph, n_ev, n_ph, channels, layers, causal, dropout)
                                    for _ in range(stages - 1))

    def forward(self, x):
        outs = [self.first(x)]
        for st in self.refine:
            o = outs[-1].detach()
            outs.append(st(torch.cat([F.softmax(o[:, :self.n_ev], 1), F.softmax(o[:, self.n_ev:], 1)], 1)))
        return outs


class SED(nn.Module):
    """Official SEDMamba BMSS blocks, our two heads in place of its classifier."""

    def __init__(self, dim_in, n_ev, n_ph, repo, num_block=3, com_factor=64, causal=False):
        super().__init__()
        sys.path.insert(0, str(repo))
        from baseline.SEDMamba import MultiStageModel

        m = MultiStageModel(num_block, com_factor, dim_in, n_ev + n_ph)
        if causal:
            for st in m.stages:
                for conv in (st.conv1, st.conv2, st.conv3):
                    d = conv.dilation[0]
                    conv.padding = (0,)
                    conv.register_forward_pre_hook(lambda mod, inp, d=d: (F.pad(inp[0], (2 * d, 0)),))
        self.m = m  # m.classify: Conv1d(dim / 2**num_block, n_ev + n_ph, 1) = our two heads

    def forward(self, x):
        return [self.m(x)]


# ---------------------------------------------------------------------------------------------- data
def video_data(p, feat_dir, label_dir, dilate_rows):
    z = np.load(Path(feat_dir) / f"{p}.npz")
    x = z["features"]
    lab = F8.VideoLabels(Path(label_dir) / p / f"{p}_all_labels.csv", fps=4)
    assert len(x) == lab.n, (p, len(x), lab.n)
    y = np.zeros(lab.n, np.int64)
    for t, k in lab.events:
        if k > 0 and F8.TARGETS[k - 1] in F8.LONG:
            c = EVENT_CLASSES.index(F8.TARGETS[k - 1])
            r = int(round(t * 4))
            y[max(0, r - dilate_rows): r + dilate_rows + 1] = c
    gts = [(EVENT_CLASSES.index(F8.TARGETS[k - 1]), float(t)) for t, k in lab.events
           if k > 0 and F8.TARGETS[k - 1] in F8.LONG and not lab.ignore[min(lab.n - 1, int(round(t * 4)))]]
    return {"x": x, "y": y, "ph": lab.phase, "m": ~lab.ignore, "second": z["second"], "gts": gts}


def losses(outs, y, ph, m, n_ev, w_ev, tmse_w):
    total = 0.0
    for o in outs:
        e, p = o[:, :n_ev].float(), o[:, n_ev:].float()
        ce_e = F.cross_entropy(e, y, weight=w_ev, reduction="none")
        ce_e = (ce_e * m).sum() / (w_ev[y] * m).sum().clamp(min=1)
        ce_p = F.cross_entropy(p, ph)
        tm = 0.0
        for logits in (e, p):
            lp = F.log_softmax(logits, 1)
            tm = tm + torch.clamp((lp[:, :, 1:] - lp.detach()[:, :, :-1]) ** 2, max=16).mean()
        total = total + ce_e + ce_p + tmse_w * tm
    return total


@torch.no_grad()
def predict(model, x, n_ev):
    o = model(x)[-1][0].float()
    return F.softmax(o[:n_ev], 0).T.cpu().numpy(), F.softmax(o[n_ev:], 0).T.cpu().numpy()


def long_map(preds, data, videos):
    dets = {v: [(c, float(data[v]["second"][t]), s) for c, t, s in
                se.decode_events(preds[v][0], se.window_frames(5.0, 4), 0.0, background_classes=(0,))]
            for v in videos}
    gts = {v: data[v]["gts"] for v in videos}
    r = se.spotting_map(dets, gts, (1, 2, 5), num_classes=len(EVENT_CLASSES), ignore_classes=(0,))
    return {d: r["map"][d] for d in (1, 2, 5)}


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=["a3", "sed"])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--label-dir", default=LAB4)
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
    ap.add_argument("--stages", type=int, default=3, help="a3: 1 Mamba stage + stages-1 MS-TCN stages")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--mamba-layers", type=int, default=4)
    ap.add_argument("--d-state", type=int, default=32)
    ap.add_argument("--headdim", type=int, default=16)
    ap.add_argument("--sed-repo", default="/home/gchie/workspace/repos/SEDMamba")
    ap.add_argument("--sed-blocks", type=int, default=3)
    ap.add_argument("--sed-g", type=int, default=64)
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
    freq = np.bincount(ys, minlength=len(EVENT_CLASSES)) / len(ys)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    w_ev = torch.tensor(w / (freq * w).sum(), dtype=torch.float32, device=dev)
    del xs, ys
    on_dev = {p: (torch.from_numpy(((d["x"].astype(np.float32) - mu) / sd).astype(np.float16)).T[None].to(dev),
                  torch.from_numpy(d["y"])[None].to(dev), torch.from_numpy(d["ph"])[None].to(dev),
                  torch.from_numpy(d["m"].astype(np.float32))[None].to(dev)) for p, d in data.items()}
    print(f"data {len(train)}/{len(inner)}/{len(held)} videos in {time.time() - t0:.0f}s, event weights "
          f"{[round(float(v), 2) for v in w_ev]}", flush=True)

    n_ev, n_ph, dim = len(EVENT_CLASSES), len(F8.PHASE_IDS), len(mu)
    if args.arch == "a3":
        model = A3(dim, n_ev, n_ph, args.stages, args.causal,
                   dict(d_model=args.d_model, layers=args.mamba_layers, d_state=args.d_state, headdim=args.headdim))
    else:
        model = SED(dim, n_ev, n_ph, args.sed_repo, args.sed_blocks, args.sed_g, args.causal)
    model = model.to(dev)
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
    cfg = dict(vars(args), n_params=n_par, event_classes=EVENT_CLASSES, mu_sd_from="train")

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
            loss = losses(model(xc), y[:, sl], ph[:, sl], m[:, sl], n_ev, w_ev, args.tmse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()
        rec = {"epoch": epoch, "train_loss": tot / len(order), "train_s": round(time.time() - te, 1)}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            preds = {p: predict(model, on_dev[p][0].float(), n_ev) for p in inner}
            mp = long_map(preds, data, inner)
            acc = float(np.mean(np.concatenate([preds[p][1].argmax(1) == data[p]["ph"] for p in inner])))
            metric = float(np.mean(list(mp.values())))
            rec.update(val_long_map={str(k): v for k, v in mp.items()}, val_metric=metric, val_phase_acc=acc)
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
        (d / "meta.json").write_text(json.dumps({"fps": 4, "event_classes": EVENT_CLASSES, "best_epoch": ck["epoch"],
                                                 "feat_dir": str(Path(args.feat_dir).resolve()), "patients": vids,
                                                 "cfg": cfg}, indent=1, default=str))
        for p in vids:
            ev, ph = predict(model, on_dev[p][0].float(), n_ev)
            np.savez_compressed(d / f"{p}.npz", second=data[p]["second"], probs_steps=ev.astype(np.float32),
                                probs_phases=ph.astype(np.float32))
    if not args.smoke:
        (out / ".done").touch()
    print(f"done: best epoch {ck['epoch']} val metric {best:.4f}, {n_par} params", flush=True)


if __name__ == "__main__":
    main()
