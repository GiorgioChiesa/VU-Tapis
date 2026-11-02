#!/usr/bin/env python3
"""
Train a light temporal model on frozen per-second clip features of full videos.

Features come from `tools/infer_untrimmed.py --features --split all`. The model is
a 2-stage MS-TCN (dilated residual 1-D convolutions) with two heads:

  * phases: dense softmax over the phase classes, cross-entropy + MS-TCN
    truncated-MSE smoothing loss;
  * events: one sigmoid per event class, trained with a focal binary
    cross-entropy against a target built from the point annotations:

      point   1 at the annotated second only
      dilate  1 on [t0 - k, t0 + k]
      gauss   exp(-(t - t0)^2 / 2 sigma^2)
      asym    linearly decaying positives on the side where the event happens and
              an ignored (unpenalised) zone of k seconds on the other side. Most
              events are stamped at their onset (positives after t0, ignore before);
              removals are stamped on the last frame the object is visible
              (positives before t0, ignore after), see OFFSET_EVENTS.

Every second of the training videos is used, so the background ("Idle") is fully
represented, including the hard seconds next to the events.

Predictions for the validation and test videos are written in the same format as
`infer_untrimmed.py` (`probs_steps`, `probs_phases`, `second`, `meta.json`), so
`tools/eval_untrimmed.py` and `tools/eval_constraints.py` evaluate them unchanged.

Example:
    python tools/train_temporal.py \
        --feat-dir outputs/orsi/LONG/control/totale/untrimmed/checkpoint_best_steps/all \
        --target asym --k 3 --out outputs/orsi/TEMPORAL/control_asym3 --gpu 0
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

from eval_constraints import read_phase_labels
from eval_untrimmed import build_label_table, global_to_local, idle_index, load_cfg
from tapis.evaluate import spotting_eval as se

# events stamped on the last frame where the object is visible (protocol: removals)
OFFSET_EVENTS = {"removing_needles", "removal_of_robotic_instruments"}


# ---- targets ---------------------------------------------------------------------
def event_targets(length, events, num_classes, kind, k=3, sigma=2.0, offset_classes=()):
    """
    Build the event target ``y`` and the loss mask ``m`` ([T, C]) of one video.

    >>> y, m = event_targets(10, [(1, 4)], 2, "asym", k=2)
    >>> (y[:, 1] * 100).round().astype(int).tolist(), m[:, 1].astype(int).tolist()
    ([0, 0, 0, 0, 100, 67, 33, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1, 1, 1, 1])
    """
    y = np.zeros((length, num_classes), dtype=np.float32)
    m = np.ones((length, num_classes), dtype=np.float32)
    t = np.arange(length)
    for c, t0 in events:
        t0 = int(round(t0))
        if not 0 <= t0 < length:
            continue
        if kind == "point":
            y[t0, c] = 1.0
        elif kind == "dilate":
            y[max(0, t0 - k) : t0 + k + 1, c] = 1.0
        elif kind == "gauss":
            y[:, c] = np.maximum(y[:, c], np.exp(-((t - t0) ** 2) / (2 * sigma**2)))
        elif kind == "asym":
            sign = -1 if c in offset_classes else 1
            for d in range(k + 1):
                ti = t0 + sign * d
                if 0 <= ti < length:
                    y[ti, c] = max(y[ti, c], 1.0 - d / (k + 1))
            for d in range(1, k + 1):
                ti = t0 - sign * d
                if 0 <= ti < length:
                    m[ti, c] = 0.0
        else:
            raise ValueError(kind)
    m[y > 0] = 1.0
    return y, m


# ---- model -------------------------------------------------------------------------
class DilatedResidual(nn.Module):
    def __init__(self, dilation, channels, causal, dropout):
        super().__init__()
        self.pad = (2 * dilation, 0) if causal else (dilation, dilation)
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.conv(F.pad(x, self.pad)))
        return x + self.drop(self.out(h))


class Stage(nn.Module):
    def __init__(self, dim_in, channels, layers, n_phases, n_events, causal, dropout):
        super().__init__()
        self.inp = nn.Conv1d(dim_in, channels, 1)
        self.layers = nn.ModuleList(DilatedResidual(2**i, channels, causal, dropout) for i in range(layers))
        self.phase = nn.Conv1d(channels, n_phases, 1)
        self.event = nn.Conv1d(channels, n_events, 1)

    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h)
        return self.phase(h), self.event(h)


class TemporalModel(nn.Module):
    def __init__(self, dim_in, n_phases, n_events, channels=64, layers=10, stages=2, causal=False, dropout=0.5):
        super().__init__()
        self.stages = nn.ModuleList(
            [Stage(dim_in, channels, layers, n_phases, n_events, causal, dropout)]
            + [Stage(n_phases + n_events, channels, layers, n_phases, n_events, causal, dropout) for _ in range(stages - 1)]
        )

    def forward(self, x):  # x: [B, D, T]
        outs = []
        for i, stage in enumerate(self.stages):
            if i:
                p, e = outs[-1]
                x = torch.cat([F.softmax(p, 1), torch.sigmoid(e)], 1).detach()
            outs.append(stage(x))
        return outs


def losses(outs, y_phase, y_event, m_event, fg, gamma, smooth_w):
    total = 0.0
    for p, e in outs:
        ce = F.cross_entropy(p, y_phase)
        logp = F.log_softmax(p, 1)
        tmse = torch.clamp((logp[:, :, 1:] - logp.detach()[:, :, :-1]) ** 2, max=16).mean()
        e, y, m = e[:, fg], y_event[:, fg], m_event[:, fg]
        prob = torch.sigmoid(e)
        bce = F.binary_cross_entropy_with_logits(e, y, reduction="none")
        focal = (torch.abs(y - prob) ** gamma) * bce * m
        ev = focal.sum() / torch.clamp(y.sum(), min=1.0)
        total = total + ce + smooth_w * tmse + ev
    return total


@torch.no_grad()
def predict(model, x, bg, fg):
    """Per-second event (sigmoid; background column = 1 - max foreground) and phase probabilities."""
    ph, ev = model(x)[-1]
    ev = torch.sigmoid(ev)[0].T.cpu().numpy()
    if bg:
        ev[:, list(bg)] = 1.0 - ev[:, fg].max(1, keepdims=True)
    return ev, F.softmax(ph, 1)[0].T.cpu().numpy()


def quick_eval(preds, data, bg, n_classes, nms_window=5.0):
    """Aggregate spotting / phase metrics on held-out videos (same definitions as eval_untrimmed)."""
    dets = {p: se.decode_events(ev, nms_window, 0.0, background_classes=bg) for p, (ev, _) in preds.items()}
    gts = {p: data[p]["events"] for p in preds}
    durations = {p: float(len(data[p]["x"])) for p in preds}
    res = se.spotting_map(dets, gts, (1, 2, 3, 4, 5, 10, 15, 20, 25, 30), n_classes, ignore_classes=bg)
    op = se.best_f1_threshold(dets, gts, 5, durations, np.round(np.arange(0.05, 1.0, 0.05), 2), bg)
    y = np.concatenate([data[p]["phase"] for p in preds])
    yhat = np.concatenate([ph.argmax(1) for _, ph in preds.values()])
    return {
        **{f"mAP@{d}s": res["map"][d] for d in (1, 2, 5, 10, 30)},
        "avg_mAP_tight": se.average_map(res["map"], se.TIGHT_DELTAS),
        "avg_mAP_loose": se.average_map(res["map"], se.LOOSE_DELTAS),
        "F1@5s": op["f1"],
        "precision@5s": op["precision"],
        "recall@5s": op["recall"],
        "FA_per_hour@5s": op["false_alarms_per_hour"],
        "phase_accuracy": float((y == yhat).mean()),
    }


# ---- main --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feat-dir", required=True)
    parser.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", default="asym", choices=["point", "dilate", "gauss", "asym"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feat-dropout", type=float, default=0.5)
    parser.add_argument("--crop", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--smooth-weight", type=float, default=0.15)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases (aggregate scalars only)")
    parser.add_argument("--wandb-project", default="OrsiTapis")
    parser.add_argument("--wandb-entity", default="eidos-giorgio")
    parser.add_argument("--wandb-group", help="default: run name without the fold suffix")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--cv-fold", help="i/n: video-level cross-validation over all patients; fold i is held out as 'val'. "
        "Only valid with features from an encoder that never saw Orsi (e.g. GraSP)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    feat_dir = Path(args.feat_dir).resolve()
    fmeta = json.loads((feat_dir / "meta.json").read_text())
    run = Path(fmeta["run"])
    cfg = load_cfg(run, fmeta["ckpt"])
    split = {
        s: [Path(f).name.removesuffix(".csv") for f in getattr(cfg.ENDOVIS_DATASET, f"{s.upper()}_LISTS")]
        for s in ("train", "val", "test")
    }
    patients = split["train"] + split["val"] + split["test"]
    if args.cv_fold:
        i, n = map(int, args.cv_fold.split("/"))
        shuffled = sorted(patients)
        random.Random(1234).shuffle(shuffled)
        held_out = set(shuffled[i::n])
        split = {
            "train": [p for p in patients if p not in held_out],
            "val": [p for p in patients if p in held_out],
            "test": [],
        }
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    table, ids = build_label_table(cfg, patients, args.frame_dir, out)
    g2l, names = global_to_local(run, "steps", ids["steps"])
    g2l_ph, ph_names = global_to_local(run, "phases", ids["phases"])
    bg = idle_index(names)
    fg = [c for c in range(len(names)) if c not in bg]
    offset = {i for i, n in enumerate(names) if n.lower() in OFFSET_EVENTS}
    events = table[(table["event_name"].str.strip().str.lower() != "idle") & table["event_id"].isin(g2l)]

    data = {}
    for p in patients:
        npz = np.load(feat_dir / f"{p}.npz")
        x = npz["features"].astype(np.float32)
        ph = read_phase_labels(args.frame_dir, p, g2l_ph)
        ev = [(g2l[r.event_id], float(r.second)) for r in events[events["patient_name"] == p].itertuples()]
        y, m = event_targets(len(x), ev, len(names), args.target, args.k, args.sigma, offset)
        data[p] = {"x": x, "phase": ph, "y": y, "m": m, "second": npz["second"], "events": ev}
    mu = np.concatenate([data[p]["x"] for p in split["train"]]).mean(0)
    sd = np.concatenate([data[p]["x"] for p in split["train"]]).std(0) + 1e-6

    model = TemporalModel(
        len(mu), len(ph_names), len(names), args.channels, args.layers, args.stages, args.causal, args.dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def tensors(p, start=0, stop=None):
        d = data[p]
        sl = slice(start, stop)
        x = torch.from_numpy((d["x"][sl] - mu) / sd).T[None].to(device)
        return (
            x,
            torch.from_numpy(d["phase"][sl])[None].long().to(device),
            torch.from_numpy(d["y"][sl]).T[None].to(device),
            torch.from_numpy(d["m"][sl]).T[None].to(device),
        )

    wb = None
    if args.wandb:
        import wandb

        name = out.name if not args.cv_fold else f"{out.parent.name}/{out.name}"
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"temporal/{name}",
            group=args.wandb_group or f"temporal/{out.parent.name if args.cv_fold else out.name}",
            job_type="temporal_head",
            dir=str(out),
            config={
                **vars(args),
                "features": str(feat_dir),
                "encoder_ckpt": fmeta.get("encoder_ckpt") or str(run / fmeta["ckpt"]),
                "n_train_videos": len(split["train"]),
                "n_val_videos": len(split["val"]),
            },
        )

    log = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(split["train"])
        random.shuffle(order)
        ep_loss = 0.0
        for p in order:
            n = len(data[p]["x"])
            start = random.randint(0, max(0, n - args.crop))
            x, yp, ye, me = tensors(p, start, start + args.crop)
            x = F.dropout1d(x, args.feat_dropout, training=True) if args.feat_dropout else x
            loss = losses(model(x), yp, ye, me, fg, args.gamma, args.smooth_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        row = {"epoch": epoch + 1, "train/loss": ep_loss / len(order), "lr": sched.get_last_lr()[0]}
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                row["val/loss"] = float(
                    np.mean(
                        [losses(model(tensors(p)[0]), *tensors(p)[1:], fg, args.gamma, args.smooth_weight).item()
                         for p in split["val"]]
                    )
                )
            preds = {p: predict(model, tensors(p)[0], bg, fg) for p in split["val"]}
            row.update({f"val/{k}": v for k, v in quick_eval(preds, data, bg, len(names)).items()})
            log.append(row)
            print(
                f"epoch {epoch + 1}: train {row['train/loss']:.4f} val {row['val/loss']:.4f} "
                f"mAP@5 {row['val/mAP@5s']:.3f} F1@5 {row['val/F1@5s']:.3f} ({time.time() - t_start:.0f}s)",
                flush=True,
            )
        if wb is not None:
            wb.log(row, step=epoch + 1)

    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "args": vars(args)}, out / "model.pt")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))

    model.eval()
    for s in ("val", "test"):
        pred_dir = out / s
        pred_dir.mkdir(exist_ok=True)
        if not split[s]:
            continue
        meta = dict(
            fmeta, split=s, patients=split[s], train_patients=split["train"], temporal_model=str(out),
            target=args.target, cv_fold=args.cv_fold,
        )
        (pred_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        preds = {p: predict(model, tensors(p)[0], bg, fg) for p in split[s]}
        for p, (ev, ph) in preds.items():
            np.savez_compressed(
                pred_dir / f"{p}.npz",
                second=data[p]["second"],
                probs_steps=ev.astype(np.float32),
                probs_phases=ph.astype(np.float32),
            )
        if wb is not None:
            wb.summary.update({f"final_{s}/{k}": v for k, v in quick_eval(preds, data, bg, len(names)).items()})
    if wb is not None:
        wb.finish()
    print(f"done: {out}")


if __name__ == "__main__":
    main()
