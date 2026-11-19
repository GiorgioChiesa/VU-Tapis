#!/usr/bin/env python3
"""
EXP-21: multi-task fine-tune of the MViT encoder at 4 fps on the 8 target events (fold 0 of the cv_finetune
folds). Heads: events (8 targets + background, soft gaussian targets, non-target events masked), phases (15),
and optionally the TAPIS teacher heads (14 actions, 7 instruments; sigmoid, soft BCE). See
tapis/datasets/focus8.py for the targets and configs/Orsi/FOCUS8/*.yaml for every parameter.

The encoder starts from a fine-tuned checkpoint (default: EXP-19 S fold 0) whose config is reused (channel
order, MViT, ORSI_AUG); its 33-class event head is replaced by the 9-class one, its phase head is kept.
Checkpoint selection uses the inner validation videos only (frame AP of the 8 targets on a fixed subsample
of rows); the held-out videos of the fold are never loaded.

Resumable: ``last.pyth`` is written after every epoch and picked up automatically.

    CUDA_VISIBLE_DEVICES=1 python tools/train_focus8.py --config configs/Orsi/FOCUS8/exp21_M.yaml \
        --out results/M/fold0 [--set KEY VALUE ...] [--smoke] [--stats-only]
"""

import argparse
import contextlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "tools", REPO / "detectron2"):
    sys.path.insert(0, str(p))

from tapis.datasets import focus8 as F  # noqa: E402

TEACHER_TASKS = ("tapis_actions", "tapis_instruments")


# ----------------------------------------------------------------------------------------------- config
def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    if len(overrides) % 2:
        raise ValueError(f"--set needs KEY VALUE pairs: {overrides}")
    for k, v in zip(overrides[::2], overrides[1::2]):
        if k not in cfg:
            raise KeyError(f"unknown config key {k}")
        cfg[k] = yaml.safe_load(v)
    return cfg


def fold_splits(cfg):
    """(train, inner_val, held_out) of fold ``cfg['fold']``: cv_finetune.folds + choose_inner_val."""
    import cv_finetune as cvf
    from eval_untrimmed import load_cfg

    ref = Path(cfg["ref_run"])
    base = yaml.safe_load(torch.load(ref / cfg["ref_ckpt"], map_location="cpu", weights_only=False)["cfg"])
    ens = base["ENDOVIS_DATASET"]
    patients = [Path(f).name.removesuffix(".csv") for k in ("TRAIN_LISTS", "VAL_LISTS", "TEST_LISTS") for f in ens[k]]
    all_folds = cvf.folds(patients)
    i = int(cfg["fold"])
    rest = sorted(p for j, f in enumerate(all_folds) if j != i for p in f)
    check = load_cfg(ref, cfg["ref_ckpt"])
    check.defrost()
    check.ENDOVIS_DATASET.FRAME_DIR = check.ENDOVIS_DATASET.ORSI_ROOT_DIR = cfg["orsi_tensors"]
    check.DATA.LABEL_DIR = cfg["split_label_dir"]
    tmp = Path(cfg["_out"]) / "split_check"
    tmp.mkdir(parents=True, exist_ok=True)
    check.OUTPUT_DIR = str(tmp)
    with contextlib.redirect_stdout(io.StringIO()):
        inner, train = cvf.choose_inner_val(rest, i, int(cfg["inner_val"]), check)
    return train, inner, all_folds[i]


# ----------------------------------------------------------------------------------------------- data
class Video:
    def __init__(self, patient, cfg, teacher):
        self.patient = patient
        lab = F.VideoLabels(Path(cfg["label_dir"]) / patient / f"{patient}_all_labels.csv", fps=cfg["fps"])
        self.n, self.phase, self.ignore, self.events = lab.n, lab.phase, lab.ignore, lab.events
        self.y, self.mask = F.event_targets(self.n, self.events, cfg["fps"], cfg["sigma_s"], cfg["nontarget_margin_s"],
                                            self.ignore)
        self.cat = F.row_categories(self.y, self.mask, self.ignore, cfg["pos_thr"])
        self.frame_path = Path(cfg["frame_dir"]) / patient / f"frames_{cfg['fps']}fps.npy"
        self.act = self.ins = None
        if teacher:
            self.act, self.ins = F.teacher_cache(patient, cfg["action_dir"], cfg["teacher_cache_dir"])


class ClipSet(torch.utils.data.Dataset):
    """Clips centred at the (video, row) pairs of ``index``; targets of all heads for the centre row."""

    def __init__(self, videos, index, cfg, aug_cfg=None):
        self.videos, self.index, self.cfg, self.aug_cfg = videos, np.asarray(index), cfg, aug_cfg
        self._arrays = {}

    def __len__(self):
        return len(self.index)

    def _frames(self, v):
        if v.patient not in self._arrays:
            self._arrays[v.patient] = np.load(v.frame_path, mmap_mode="r")
        return self._arrays[v.patient]

    def __getitem__(self, i):
        vi, r = (int(x) for x in self.index[i])
        v = self.videos[vi]
        rows = F.clip_rows(r, v.n, self.cfg["num_frames"])
        arr = self._frames(v)
        assert len(arr) == v.n, (v.patient, len(arr), v.n)
        x = torch.from_numpy(np.ascontiguousarray(arr[rows])).float() / 255.0  # T,C,H,W (stored channel order)
        if self.aug_cfg is not None:
            from tapis.datasets import clip_augment

            p = clip_augment.sample_params(self.aug_cfg, x.shape[-2], x.shape[-1])
            x = clip_augment.apply(x, p)
        out = {"x": x.permute(1, 0, 2, 3).contiguous(), "y_event": torch.from_numpy(v.y[r]),
               "m_event": bool(v.mask[r]), "phase": int(v.phase[r]), "video": vi, "row": r}
        if v.act is not None:
            a, ok_a = F.teacher_window(v.act, rows, self.cfg["fps"])
            s, ok_s = F.teacher_window(v.ins, rows, self.cfg["fps"])
            out.update(y_act=torch.from_numpy(a), m_act=ok_a, y_ins=torch.from_numpy(s), m_ins=ok_s)
        return out


def val_rows(v, cfg):
    """Fixed inner-val rows: every row within ``val_near_s`` of a target event + every ``val_stride_s`` s."""
    t = np.arange(v.n) / cfg["fps"]
    keep = np.zeros(v.n, bool)
    keep[:: int(cfg["val_stride_s"] * cfg["fps"])] = True
    for te, k in v.events:
        if k > 0:
            keep |= np.abs(t - te) <= cfg["val_near_s"]
    keep &= ~v.ignore
    return np.flatnonzero(keep)


def frame_ap(y, mask, prob, thr=0.5):
    """Per-target frame AP (positive = soft target >= thr) over masked rows; NaN for a class without positives."""
    from sklearn.metrics import average_precision_score

    out = []
    for k in range(1, y.shape[1]):
        pos = y[mask, k] >= thr
        out.append(float(average_precision_score(pos, prob[mask, k])) if pos.any() else float("nan"))
    return out


# ----------------------------------------------------------------------------------------------- model
def build(cfg, teacher, device):
    from fvcore.common.config import CfgNode

    from tapis.config.defaults import get_cfg
    from tapis.models import build_model

    ck = torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=False)
    mcfg = get_cfg()
    mcfg.set_new_allowed(True)
    mcfg.merge_from_other_cfg(CfgNode.load_cfg(ck["cfg"]))
    tasks = ["steps", "phases"] + (list(TEACHER_TASKS) if teacher else [])
    ncls = [len(F.EVENT_CLASSES), len(F.PHASE_IDS)] + ([len(F.ACTIONS), len(F.INSTRUMENTS)] if teacher else [])
    mcfg.TASKS.TASKS = tasks
    mcfg.TASKS.NUM_CLASSES = ncls
    mcfg.TASKS.HEAD_ACT = ["softmax", "softmax"] + (["sigmoid", "sigmoid"] if teacher else [])
    mcfg.TASKS.LOSS_FUNC = ["cross_entropy", "cross_entropy"] + (["bce", "bce"] if teacher else [])
    mcfg.TASKS.METRICS = ["mAP"] * len(tasks)
    mcfg.TASKS.LOSS_WEIGHTS = [cfg["w_events"], cfg["w_phases"]] + ([cfg["w_actions"], cfg["w_instruments"]] if teacher else [])
    mcfg.TASKS.WEIGHT_LOSS_BY_CLASS = [""] * len(tasks)
    mcfg.NUM_GPUS = 1
    mcfg.DATA.LABEL_DIR = cfg["label_dir"]
    mcfg.DATA.FRAME_DIR = cfg["frame_dir"]
    mcfg.DATA.FPS = cfg["fps"]
    assert mcfg.DATA.NUM_FRAMES == cfg["num_frames"] and mcfg.DATA.SEQ_MODE == "center"
    model = build_model(mcfg, gpu_id=device)
    state = {k: v for k, v in ck["model_state"].items() if not k.startswith("extra_heads_steps.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    new = [m for m in missing if not m.startswith("extra_heads_")]
    assert not new and not unexpected, (new[:5], unexpected[:5])
    print(f"init {cfg['init_ckpt']}: new heads {sorted({m.split('.')[0] for m in missing})}", flush=True)
    return mcfg, model


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total, warmup, base, end):
    if step < warmup:
        return base * (0.1 + 0.9 * step / max(1, warmup))
    q = (step - warmup) / max(1, total - warmup)
    return end + (base - end) * 0.5 * (1 + math.cos(math.pi * min(1.0, q)))


def class_weights(videos, w_rows, mode, max_ratio=10.0):
    if mode == "none":
        return torch.ones(len(F.EVENT_CLASSES))
    y = np.concatenate([v.y for v in videos])
    m = np.concatenate([v.mask for v in videos])
    freq = (w_rows[m, None] * y[m]).sum(0)
    freq = freq / freq.sum()
    w = np.minimum(1 / np.sqrt(np.maximum(freq, 1e-6)), max_ratio / np.sqrt(freq.max()))  # absent class: capped
    w = w / (freq * w).sum()
    return torch.tensor(w, dtype=torch.float32)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true", help="2 train videos, 1 inner-val video, 1 short epoch")
    ap.add_argument("--stats-only", action="store_true", help="write data_stats.json and exit (no GPU)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, args.set)
    cfg["_out"] = str(out)
    teacher = bool(cfg["teacher"])
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    split_file = out / "splits.json"
    if split_file.exists():
        sp = json.loads(split_file.read_text())
    else:
        train, inner, held = fold_splits(cfg)
        sp = {"train": train, "inner_val": inner, "held_out": held}
        split_file.write_text(json.dumps(sp, indent=1))
    print(f"fold {cfg['fold']}: {len(sp['train'])} train, inner-val {sp['inner_val']}, "
          f"{len(sp['held_out'])} held out (never loaded)", flush=True)
    train_p, val_p = sp["train"], sp["inner_val"]
    if args.smoke:
        train_p, val_p = train_p[:2], val_p[:1]
        cfg.update(clips_per_epoch=int(cfg["smoke_clips"]), epochs=1, val_stride_s=60)

    t0 = time.time()
    tv = [Video(p, cfg, teacher) for p in train_p]
    vv = [Video(p, cfg, teacher) for p in val_p]
    cats = np.concatenate([v.cat for v in tv])
    w_rows = F.sampling_weights(cats, cfg["share_target"], cfg["share_background"], cfg["share_nontarget"])
    offs = np.cumsum([0] + [v.n for v in tv])
    vid_of = np.repeat(np.arange(len(tv)), [v.n for v in tv])

    def count_targets(vs):
        c = np.zeros(len(F.TARGETS), int)
        for v in vs:
            for _, k in v.events:
                if k > 0:
                    c[k - 1] += 1
        return dict(zip(F.TARGETS, c.tolist()))

    stats = {
        "label_dir": cfg["label_dir"], "train_videos": len(tv), "inner_val_videos": len(vv),
        "target_events_train": count_targets(tv), "target_events_inner_val": count_targets(vv),
        "rows_train": int(len(cats)), "rows_by_category": {str(k): int((cats == k).sum()) for k in range(-2, 9)},
        "event_mask_fraction_train": float(np.concatenate([v.mask for v in tv]).mean()),
    }
    if teacher:
        stats["teacher_seconds_valid_fraction"] = float(np.mean(np.concatenate(
            [np.isfinite(v.act).all(1) for v in tv])))
    (out / ("data_stats_smoke.json" if args.smoke else "data_stats.json")).write_text(json.dumps(stats, indent=1))
    print(f"data loaded in {time.time() - t0:.0f}s: {json.dumps(stats)}", flush=True)
    if args.stats_only:
        return

    device = 0
    torch.cuda.set_device(device)
    mcfg, model = build(cfg, teacher, device)
    aug = mcfg.ORSI_AUG if (cfg["augment"] and mcfg.ORSI_AUG.ENABLE) else None
    cw = class_weights(tv, w_rows, cfg["event_class_weight"]).cuda()
    pw = torch.ones(len(F.INSTRUMENTS)).cuda()
    nw = torch.ones(len(F.INSTRUMENTS)).cuda()
    for name, w in (cfg.get("ins_pos_weight") or {}).items():
        pw[F.INSTRUMENTS.index(name)] = float(w)
    for name, w in (cfg.get("ins_neg_weight") or {}).items():
        nw[F.INSTRUMENTS.index(name)] = float(w)
    print(f"event class weights {[round(float(x), 3) for x in cw]}", flush=True)

    opt = torch.optim.Adam(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    steps_per_epoch = math.ceil(cfg["clips_per_epoch"] / (cfg["batch_size"] * cfg["accum_steps"]))
    total = steps_per_epoch * cfg["epochs"]
    warm = int(cfg["warmup_epochs"] * steps_per_epoch)
    start_epoch, best, step = 0, -1.0, 0
    last = out / "last.pyth"
    if last.exists() and not args.smoke:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        start_epoch, best, step = ck["epoch"] + 1, ck["best"], ck["step"]
        print(f"resumed from {last}: epoch {ck['epoch']} done, best {best:.4f}", flush=True)

    vidx = np.concatenate([np.stack([np.full(len(r), i), r], 1) for i, v in enumerate(vv) for r in [val_rows(v, cfg)]])
    val_set = ClipSet(vv, vidx, cfg)
    dl_kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True)
    log = open(out / ("log_smoke.jsonl" if args.smoke else "log.jsonl"), "a")

    def save(path, epoch, extra=None):
        ck = {"model_state": model.state_dict(), "cfg": mcfg.dump(), "epoch": epoch, "best": best, "step": step,
              "focus8": {"config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "splits": sp,
                         "event_classes": F.EVENT_CLASSES, "phase_ids": F.PHASE_IDS, "actions": F.ACTIONS,
                         "instruments": F.INSTRUMENTS}}
        ck.update(extra or {})
        tmp = path.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(path)

    for epoch in range(start_epoch, cfg["epochs"]):
        rng = np.random.default_rng(seed * 1000 + epoch)
        rows = rng.choice(len(w_rows), size=int(cfg["clips_per_epoch"]), p=w_rows)
        idx = np.stack([vid_of[rows], rows - offs[vid_of[rows]]], 1)
        g = torch.Generator()
        g.manual_seed(seed * 1000 + epoch)
        dl = torch.utils.data.DataLoader(ClipSet(tv, idx, cfg, aug), shuffle=True, drop_last=True, generator=g,
                                         **dl_kw)
        model.train()
        sums, nb, te = {}, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(dl):
            x = b["x"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                o = model([x])
            losses = {"events": F.soft_ce(o["steps"], b["y_event"].cuda(), b["m_event"].cuda(), cw),
                      "phases": torch.nn.functional.cross_entropy(o["phases"].float(), b["phase"].cuda())}
            if teacher:
                losses["actions"] = F.asymmetric_bce(o["tapis_actions"], b["y_act"].cuda(), b["m_act"].cuda())
                losses["instruments"] = F.asymmetric_bce(o["tapis_instruments"], b["y_ins"].cuda(), b["m_ins"].cuda(),
                                                         pw, nw)
            loss = sum(cfg[f"w_{k}"] * v for k, v in losses.items())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} iter {it}: "
                                         f"{ {k: float(v) for k, v in losses.items()} }")
            scaler.scale(loss / cfg["accum_steps"]).backward()
            if (it + 1) % cfg["accum_steps"] == 0:
                for gp in opt.param_groups:
                    gp["lr"] = lr_at(step, total, warm, cfg["lr"], cfg["end_lr"])
                scaler.unscale_(opt)
                if cfg["clip_grad_l2norm"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_l2norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            sums["total"] = sums.get("total", 0.0) + float(loss)
            nb += 1
            if it % cfg["log_every"] == 0:
                print(f"ep {epoch} it {it}/{len(dl)} " + " ".join(f"{k} {v / nb:.4f}" for k, v in sums.items())
                      + f" lr {opt.param_groups[0]['lr']:.2e} {(it + 1) * cfg['batch_size'] / (time.time() - te):.1f} clips/s",
                      flush=True)
        train_s = time.time() - te

        # inner validation
        model.eval()
        tv0 = time.time()
        probs = {"steps": [], "phases": []}
        ys, ms, ph = [], [], []
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(val_set, shuffle=False, **dl_kw):
                with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                    o = model([b["x"].cuda(non_blocking=True)])
                probs["steps"].append(o["steps"].float().cpu())
                probs["phases"].append(o["phases"].float().cpu())
                ys.append(b["y_event"])
                ms.append(b["m_event"])
                ph.append(b["phase"])
        p_ev = torch.cat(probs["steps"]).numpy()
        y, m = torch.cat(ys).numpy(), torch.cat(ms).numpy().astype(bool)
        aps = frame_ap(y, m, p_ev)
        short, long_ = np.nanmean(aps[:4]), np.nanmean(aps[4:])
        metric = float(np.nanmean(aps))
        ph_acc = float((torch.cat(probs["phases"]).argmax(1) == torch.cat(ph)).float().mean())
        rec = {"epoch": epoch, "step": step, "train_s": round(train_s, 1), "val_s": round(time.time() - tv0, 1),
               **{f"loss_{k}": v / max(1, nb) for k, v in sums.items()},
               "val_event_frame_map": metric, "val_short_frame_map": float(short), "val_long_frame_map": float(long_),
               "val_ap_per_class": dict(zip(F.TARGETS, aps)), "val_phase_acc": ph_acc, "val_rows": int(len(val_set))}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"epoch {epoch}: {json.dumps(rec)}", flush=True)
        if args.smoke:
            break
        if metric > best:
            best = metric
            save(out / "best.pyth", epoch)
        save(last, epoch, {"optimizer_state": opt.state_dict(), "scaler_state": scaler.state_dict()})
        if cfg["keep_every"] and (epoch + 1) % cfg["keep_every"] == 0:
            save(out / f"epoch_{epoch:03d}.pyth", epoch)
    if not args.smoke:
        (out / ".done").touch()
    print("done", flush=True)


if __name__ == "__main__":
    main()
