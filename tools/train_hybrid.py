#!/usr/bin/env python3
"""
Hybrid model: frozen per-second clip features + an interchangeable global-context module.

Same data, targets, losses, folds and outputs as `tools/train_temporal.py` (imported,
not copied); the temporal model is any context module of
`tapis/models/temporal_context.py` (G0..G6) followed by the phase and event heads:

  G0 per-second MLP | G1 local TCN (1..32) | G2 MS-TCN (= train_temporal) | G3 (Bi)GRU
  G4 transformer on 8x tokens | G5 G1 + G4 | G6 LSTR-style memory (causal only)

With the same seed, `--context G2` consumes the random streams in the same order as
`train_temporal.py`, so `cv_grasp_dilate3` is reproduced run by run.

Outputs, in --out:
  model.pt, train_log.json
  val/<patient>.npz + val/meta.json   readable by eval_untrimmed.py / eval_constraints.py
  val/hybrid_metrics.json             quick spotting/phase metrics, segmental phase metrics
                                      (edit, F1@10/25/50), parameters, train time, latency

Experiment 2 hooks (parsed, not implemented yet): --datasets, --aux-mode, --aux-sources,
--phase-cond, --oracle-phase, --streams. `load_orsi` is the dataset adapter; other
datasets plug in next to it and `HybridModel.aux` holds their heads.

Example:
    python tools/train_hybrid.py --context G4 --causal --target dilate --k 3 --cv-fold 0/5 \
        --feat-dir outputs/orsi/LONG/control/totale/untrimmed/grasp_LONG/all \
        --out outputs/orsi/HYBRID/G4c_dilate3/seed0/fold0 --gpu 2
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
from train_temporal import OFFSET_EVENTS, event_targets, losses, predict, quick_eval
from tapis.evaluate import segment_eval
from tapis.models.temporal_context import CONTEXTS, build_context, count_parameters


# ---- data --------------------------------------------------------------------------
def cv_split(patients, fold):
    """Video-level fold ``i/n`` (same shuffle as train_temporal.py)."""
    i, n = map(int, fold.split("/"))
    shuffled = sorted(patients)
    random.Random(1234).shuffle(shuffled)
    held_out = set(shuffled[i::n])
    return {
        "train": [p for p in patients if p not in held_out],
        "val": [p for p in patients if p in held_out],
        "test": [],
    }


def load_orsi(args, out):
    """
    Orsi adapter: per-patient dict ``{x, phase, y, m, second, events}`` plus the split and
    the class vocabularies. Mirrors train_temporal.main (same tables, same order).
    """
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
        split = cv_split(patients, args.cv_fold)

    table, ids = build_label_table(cfg, patients, args.frame_dir, out)
    g2l, names = global_to_local(run, "steps", ids["steps"])
    g2l_ph, ph_names = global_to_local(run, "phases", ids["phases"])
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
    info = {"feat_dir": feat_dir, "fmeta": fmeta, "run": run, "names": names, "ph_names": ph_names}
    return data, split, info


DATASETS = {"orsi": load_orsi}


# ---- model -------------------------------------------------------------------------
class HybridModel(nn.Module):
    """
    Context module + phase/event heads, with the calling convention of
    train_temporal.TemporalModel: ``x [B, D, T] -> [(phase [B, P, T], event [B, E, T]), ...]``
    (one pair per supervised stage, the last one is the prediction).

    Hooks for Experiment 2: ``aux`` (extra heads per auxiliary dataset/task, `--aux-mode
    joint`) and ``phase_cond`` (event head conditioned on the phase posterior).
    """

    def __init__(self, context, dim_in, n_phases, n_events, causal=False, **kwargs):
        super().__init__()
        self.n_phases = n_phases
        self.context = build_context(context, dim_in, n_phases + n_events, causal=causal, n_phases=n_phases, **kwargs)
        self.aux = nn.ModuleDict()
        self.phase_cond = None

    def forward(self, x):
        outs = self.context.forward_all(x.transpose(1, 2))
        return [(o[..., : self.n_phases].transpose(1, 2), o[..., self.n_phases :].transpose(1, 2)) for o in outs]


@torch.no_grad()
def latency(model, x, device, repeats=3):
    """Seconds of GPU compute to process ``x`` (one full video), best of ``repeats``."""
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        model(x)
        torch.cuda.synchronize(device)
        best = min(best, time.perf_counter() - t)
    return best


# ---- main --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", required=True, choices=CONTEXTS)
    parser.add_argument("--feat-dir", required=True)
    parser.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", default="dilate", choices=["point", "dilate", "gauss", "asym"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feat-dropout", type=float, default=0.5)
    parser.add_argument("--crop", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--smooth-weight", type=float, default=0.15)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--cv-fold", help="i/n: video-level CV over all patients (GraSP features only)")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases (aggregate scalars only)")
    parser.add_argument("--wandb-project", default="OrsiTapis")
    parser.add_argument("--wandb-entity", default="eidos-giorgio")
    parser.add_argument("--wandb-group", help="default: hybrid/<config> (run dir without seed/fold)")
    # ---- Experiment 2 (hooks only) ----
    parser.add_argument("--datasets", nargs="+", default=["orsi"], help="training datasets (Exp. 2)")
    parser.add_argument("--aux-mode", default="none", choices=["none", "joint", "stack"])
    parser.add_argument("--aux-sources", nargs="*", default=[])
    parser.add_argument("--phase-cond", action="store_true", help="event head conditioned on the phase posterior")
    parser.add_argument("--oracle-phase", action="store_true", help="ground-truth phases as input (upper bound)")
    parser.add_argument("--streams", default="dynamic", choices=["dynamic", "static", "two"])
    args = parser.parse_args(argv)
    todo = [
        f for f, on in (
            ("--datasets", args.datasets != ["orsi"]), ("--aux-mode", args.aux_mode != "none"),
            ("--aux-sources", bool(args.aux_sources)), ("--phase-cond", args.phase_cond),
            ("--oracle-phase", args.oracle_phase), ("--streams", args.streams != "dynamic"),
        ) if on
    ]
    if todo:
        raise NotImplementedError(f"Experiment 2 options not implemented yet: {todo}")
    return args


def run_name(out, cv_fold):
    """``<config>/seed<s>/fold<i>`` -> W&B run name and default group."""
    parts = out.parts[-3:] if cv_fold else out.parts[-1:]
    group = parts[0]
    return "/".join(parts), group


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, split, info = DATASETS[args.datasets[0]](args, out)
    names, ph_names = info["names"], info["ph_names"]
    bg = idle_index(names)
    fg = [c for c in range(len(names)) if c not in bg]

    mu = np.concatenate([data[p]["x"] for p in split["train"]]).mean(0)
    sd = np.concatenate([data[p]["x"] for p in split["train"]]).std(0) + 1e-6

    model = HybridModel(args.context, len(mu), len(ph_names), len(names), args.causal, dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"{args.context}{' causal' if args.causal else ''}: {n_params / 1e6:.3f} M parameters", flush=True)
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

    name, group = run_name(out, args.cv_fold)
    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"hybrid/{name}",
            group=args.wandb_group or f"hybrid/{group}",
            job_type="hybrid",
            dir=str(out),
            config={
                **vars(args),
                "features": str(info["feat_dir"]),
                "encoder_ckpt": info["fmeta"].get("encoder_ckpt") or str(info["run"] / info["fmeta"]["ckpt"]),
                "n_params": n_params,
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
    train_time = time.time() - t_start

    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "args": vars(args)}, out / "model.pt")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))

    model.eval()
    for s in ("val", "test"):
        pred_dir = out / s
        pred_dir.mkdir(exist_ok=True)
        if not split[s]:
            continue
        meta = dict(
            info["fmeta"], split=s, patients=split[s], train_patients=split["train"], temporal_model=str(out),
            target=args.target, cv_fold=args.cv_fold, context=args.context, causal=args.causal,
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
        longest = max(split[s], key=lambda p: len(data[p]["x"]))
        lat = latency(model, tensors(longest)[0], device)
        metrics = {
            **quick_eval(preds, data, bg, len(names)),
            **{f"phase_{k}": v for k, v in segment_eval.segmental_metrics(
                {p: ph.argmax(1) for p, (_, ph) in preds.items()}, {p: data[p]["phase"] for p in preds}
            ).items()},
            "n_params": n_params,
            "train_time_s": train_time,
            "latency_ms_per_video_hour": 1000 * lat / (len(data[longest]["x"]) / 3600),
        }
        (pred_dir / "hybrid_metrics.json").write_text(json.dumps(metrics, indent=2))
        if wb is not None:
            wb.summary.update({f"final_{s}/{k}": v for k, v in metrics.items()})
    if wb is not None:
        wb.finish()
    print(f"done: {out}")


if __name__ == "__main__":
    main()
