#!/usr/bin/env python3
"""
Run a trained TAPIS checkpoint on every second of full (untrimmed) Orsi videos.

For each second of a video the clip is built exactly as in training/validation
(`Orsi.__getitem__`): `utils.get_sequence` with the checkpoint's DATA.SEQ_MODE,
NUM_FRAMES and SAMPLING_RATE over the rows of `{patient}_all_labels.csv`, frames
loaded from the pre-computed `Video_1fps/*.pt` tensors and fed as CTHW in fp32.

Outputs one `{patient}.npz` per video with the per-second probabilities of every
task (`probs_steps`, `probs_phases`, ...) plus `second` and `frame_id`. These
files are derived from patient data: keep them on this machine (outputs/ is
gitignored).

Example:
    python tools/infer_untrimmed.py --run outputs/orsi/LONG/control/totale \
        --ckpt checkpoint_best_steps.pyth --split val \
        --frame-dir /home/gchie/workspace/data/orsi_tensors
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "detectron2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch
from fvcore.common.config import CfgNode

from tapis.config.defaults import get_cfg
from tapis.datasets import utils as data_utils
from tapis.models import build_model


def load_cfg_and_model(ckpt_path, gpu, encoder_ckpt=None):
    """
    Build the model from the config stored in ``ckpt_path``. With ``encoder_ckpt``
    the weights come from that file instead (e.g. the official GraSP-trained TAPIS,
    which has no ``cfg``): its task heads are read from the state dict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # older checkpoints carry keys removed from defaults.py
    cfg.merge_from_other_cfg(CfgNode.load_cfg(ckpt["cfg"]))
    state = ckpt["model_state"]
    if encoder_ckpt is not None:
        state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)["model_state"]
        heads = {
            k.split(".")[0].removeprefix("extra_heads_"): v.shape[0]
            for k, v in state.items()
            if k.startswith("extra_heads_") and k.endswith("class_projection.weight")
        }
        cfg.TASKS.TASKS = list(heads)
        cfg.TASKS.NUM_CLASSES = list(heads.values())
        cfg.TASKS.LOSS_FUNC = [cfg.TASKS.LOSS_FUNC[0]] * len(heads)
        cfg.TASKS.HEAD_ACT = [cfg.TASKS.HEAD_ACT[0]] * len(heads)
    cfg.NUM_GPUS = 1
    torch.cuda.set_device(gpu)
    model = build_model(cfg, gpu_id=gpu)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")
    model.eval()
    return cfg, model


def patients_for_split(cfg, split):
    lists = {
        "train": cfg.ENDOVIS_DATASET.TRAIN_LISTS,
        "val": cfg.ENDOVIS_DATASET.VAL_LISTS,
        "test": cfg.ENDOVIS_DATASET.TEST_LISTS,
    }
    lists["all"] = list(lists["train"]) + list(lists["val"]) + list(lists["test"])
    lists = lists[split]
    return [os.path.basename(f).removesuffix(".csv").removesuffix("_all_label") for f in lists]


def load_frames(frame_dir, patient, frame_paths, workers=16):
    """Load all 1 fps tensors of a video into one [N, 3, H, W] float32 tensor."""

    def _one(rel):
        t = torch.load(
            os.path.join(frame_dir, patient, "Video_1fps", os.path.basename(rel)),
            map_location="cpu",
            weights_only=True,
        )
        # Training goes through utils._load_tensor, which only keeps CHW float32
        # tensors as they are; anything else would be preprocessed differently.
        assert t.dtype == torch.float32 and t.ndim == 3 and t.shape[0] == 3, (rel, t.dtype, t.shape)
        return t

    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(_one, frame_paths))
    return torch.stack(frames)


@torch.no_grad()
def infer_video(cfg, model, frames, batch_size, gpu, amp, features=False):
    """
    Per-second task probabilities; with ``features=True`` also the clip embedding
    the heads read (CLS token(s), flattened) concatenated with the mean patch token.
    """
    n = frames.shape[0]
    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    seqs = torch.tensor(
        [
            data_utils.get_sequence(i, seq_len, cfg.DATA.SAMPLING_RATE, num_frames=n, mode=cfg.DATA.SEQ_MODE)
            for i in range(n)
        ]
    )
    frames = frames.cuda(gpu, non_blocking=True)
    if hasattr(model, "reset_memory_bank"):
        model.reset_memory_bank()
    out = {task: [] for task in cfg.TASKS.TASKS}
    if features:
        out["features"] = []
    for start in range(0, n, batch_size):
        idx = seqs[start : start + batch_size].cuda(gpu)
        clips = frames[idx].permute(0, 2, 1, 3, 4).contiguous()  # B,T,C,H,W -> B,C,T,H,W
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            preds = model([clips])
        if isinstance(preds, tuple):
            preds = preds[0]
        for task in cfg.TASKS.TASKS:
            out[task].append(preds[task].float().cpu())
        if features:
            cls = preds["cls_tokens"]
            tokens = preds["features"][:, 0 if cls is None else cls.shape[1] :]
            parts = ([] if cls is None else [cls.flatten(1)]) + [tokens.mean(1)]
            out["features"].append(torch.cat(parts, dim=1).half().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run output dir containing the checkpoint")
    parser.add_argument("--ckpt", default="checkpoint_best_steps.pyth")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--patients", nargs="*", help="Override the patients of --split")
    parser.add_argument("--frame-dir", default="/data/orsi_tensors")
    parser.add_argument("--out", help="Default: <run>/untrimmed/<ckpt stem>/<split>")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="fp16 autocast (validation in train_net runs in fp32)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--features", action="store_true", help="also save the clip embeddings (fp16)")
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th patient starting at i")
    parser.add_argument("--encoder-ckpt", help="take the weights from this file (cfg still from --run/--ckpt)")
    parser.add_argument("--normalize", action="store_true", help="(x - DATA.MEAN) / DATA.STD, as in GraSP training")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    cfg, model = load_cfg_and_model(run / args.ckpt, args.gpu, args.encoder_ckpt)
    patients = args.patients or patients_for_split(cfg, args.split)
    shard, n_shards = map(int, args.shard.split("/"))
    out_dir = Path(args.out) if args.out else run / "untrimmed" / Path(args.ckpt).stem / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run": str(run),
        "ckpt": args.ckpt,
        "split": args.split,
        "patients": patients,
        "tasks": list(cfg.TASKS.TASKS),
        "num_classes": list(cfg.TASKS.NUM_CLASSES),
        "seq_mode": cfg.DATA.SEQ_MODE,
        "num_frames": cfg.DATA.NUM_FRAMES,
        "sampling_rate": cfg.DATA.SAMPLING_RATE,
        "amp": args.amp,
    }
    meta["features"] = args.features
    meta["encoder_ckpt"] = args.encoder_ckpt
    meta["normalize"] = args.normalize
    if shard == 0:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    for patient in patients[shard::n_shards]:
        target = out_dir / f"{patient}.npz"
        if target.exists() and not args.overwrite:
            print(f"{patient}: exists, skipping", flush=True)
            continue
        t0 = time.time()
        labels = pd.read_csv(os.path.join(args.frame_dir, patient, "Label", f"{patient}_all_labels.csv"))
        frames = load_frames(args.frame_dir, patient, labels["frame_path"].tolist())
        if args.normalize:
            mean = torch.tensor(cfg.DATA.MEAN).view(1, 3, 1, 1)
            std = torch.tensor(cfg.DATA.STD).view(1, 3, 1, 1)
            frames = (frames - mean) / std
        t1 = time.time()
        probs = infer_video(cfg, model, frames, args.batch_size, args.gpu, args.amp, args.features)
        del frames
        torch.cuda.empty_cache()
        feats = probs.pop("features", None)
        for task, p in probs.items():
            row_sums = p.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-3), f"{task} outputs are not probabilities"
        np.savez_compressed(
            target,
            second=labels["second"].to_numpy(),
            frame_id=labels["frame_id"].to_numpy(),
            **{f"probs_{task}": p.astype(np.float32) for task, p in probs.items()},
            **({"features": feats} if feats is not None else {}),
        )
        print(
            f"{patient}: {len(labels)} s, load {t1 - t0:.0f}s, infer {time.time() - t1:.0f}s "
            f"({len(labels) / max(time.time() - t1, 1e-6):.1f} clips/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
