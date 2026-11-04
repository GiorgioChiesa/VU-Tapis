#!/usr/bin/env python3
"""
Experiment 5: clip features at sub-second (0.25 s grid) positions, straight from the Orsi mp4s.

The 1 fps features of `tools/infer_untrimmed.py` embed, for second ``i``, the clip of the 16
`Video_1fps` frames at seconds ``i-8 .. i+7`` (`utils.get_sequence`, mode center, sampling rate 1,
indices clamped to ``[0, n-1]``). Here a position ``p`` (any multiple of 0.25 s) gets the clip of
the frames at times ``p-8 .. p+7`` (1 s apart, times clamped to ``[0, n-1]`` s), so that ``p = i``
gives back exactly the 1 fps clip. Frames are rebuilt as `data/check_webapp/main.py` built
`Video_1fps` (the script that wrote them):

  * frame of time ``t`` = the frame ``cap.set(CAP_PROP_POS_MSEC, t * 1000); cap.read()`` returns.
    44 of the 50 videos are variable frame rate, where that is *not* frame ``round(t * fps)``:
    OpenCV seeks to frame number ``N = floor(t * fps + 0.5)`` (``fps`` = average rate) by jumping to
    the last keyframe whose pts <= ``(N - 16) / fps`` and counting frames from the keyframe's
    pts-derived number. `SeekEmulator` reproduces this from the packet table (`ffprobe`, no
    decoding; decoding order = packets sorted by pts, dts), so the video is decoded only once;
  * ``to_pil_image`` of the BGR array, ``removeBlackBorders``, ``scale(224)``, ``center_crop(224)``,
    ``pil_to_tensor / 255`` (the functions are imported from `data/`, not re-implemented).

Each video is decoded once, sequentially (cv2, CPU worker processes); only the needed frames are
kept, as uint8, in a rolling RAM buffer (nothing is written to disk but the features). Groups of
positions travel with the frames they need to one GPU worker per ``--gpus`` entry, which runs the
encoder of ``--run`` (features = CLS token(s) flattened + mean patch token, as `infer_video`).

Output per video: ``<out>/<patient>.npz`` with ``positions`` [N] (s, float32, sorted) and
``features`` [N, 1536] fp16. Positions come from ``<positions-dir>/<patient>.npy`` (seconds).

``--check-frames`` compares every rebuilt integer-second frame with its ``Video_1fps/*.pt``;
``--check-features DIR`` compares the features at integer positions with ``DIR/<patient>.npz`` of
`infer_untrimmed.py --features`. Results (aggregate) go to ``<out>/fidelity.json``. Expected:
every frame identical; cosine ~1 except within 8 s of a second that the CSV lists twice (two events
on one second), where the 1 fps clip, built over CSV rows, holds that frame twice.

Example:
    python tools/extract_shifted_features.py --run outputs/orsi/LONG/cv_ft/fold0 \
        --positions-dir outputs/orsi/EXP5/positions/fold0 --out outputs/orsi/EXP5/feat/fold0 \
        --gpus 2 3 --workers 16
    python tools/extract_shifted_features.py --run outputs/orsi/LONG/cv_ft/fold0 --patients RARP01 RARP30 \
        --integer-positions --check-frames --out outputs/orsi/EXP5/fidelity/fold0 --gpus 2 3 \
        --check-features outputs/orsi/LONG/cv_ft/fold0/untrimmed/checkpoint_best_steps/all
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "detectron2", REPO / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

DATA_DIR = "/home/gchie/workspace/data"
VIDEO_DIR = Path("/home/gchie/workspace/nas_private/data/orsi/videos")
FRAME_DIR = Path("/home/gchie/workspace/data/orsi_tensors")
GRID = 4  # positions per second
OFFSETS = range(-8, 8)  # clip frames, seconds from the position (NUM_FRAMES 16, center, rate 1)
MIN_FREE_GB = 100


def clip_grid(g, n_seconds):
    """
    Grid indices (1/GRID s) of the 16 clip frames of the position at grid index ``g``,
    clamped to ``[0, n_seconds - 1]`` s like `utils.get_sequence`.

    >>> clip_grid(40, 100)[:3], clip_grid(40, 100)[-1]      # p = 10 s: 2, 3, 4 ... 17 s
    ([8, 12, 16], 68)
    >>> clip_grid(5, 100)[:3], clip_grid(5, 100)[8]          # p = 1.25 s: clamped to 0 s
    ([0, 0, 0], 5)
    >>> clip_grid(398, 100)[-3:]                             # p = 99.5 s: last frame 99 s
    [396, 396, 396]
    """
    hi = (n_seconds - 1) * GRID
    return [min(max(g + GRID * k, 0), hi) for k in OFFSETS]


class SeekEmulator:
    """
    Decoded frame (index in presentation order) that OpenCV 4.x returns after
    ``cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)`` (`CvCapture_FFMPEG::seek`), from the packet table.

    ``pts`` / ``dts`` / ``key`` are the packets in any order, ``tb`` the stream time base (s),
    ``fps`` the average frame rate (``CAP_PROP_FPS``). With a constant frame rate it is frame
    ``floor(t * fps + 0.5)``; with drops between keyframes it follows OpenCV's frame counting:

    >>> pts = [i * 1001 for i in range(300)]                   # CFR 59.94, keyframes every 100 frames
    >>> em = SeekEmulator(pts, pts, [i % 100 == 0 for i in range(300)], 1 / 60000, 60000 / 1001)
    >>> em.frame(1.0), em.frame(2.0), em.frame(0.0), em.frame(0.25)
    (60, 120, 0, 15)
    >>> pts = [i * 1001 for i in range(10)] + [i * 1001 for i in range(15, 300)]   # 5 frames dropped
    >>> em = SeekEmulator(pts, pts, [p % 100100 == 0 for p in pts], 1 / 60000, 60000 / 1001)
    >>> em.frame(3.0)          # keyframe pts 2.002 s = frame number 120, but it is the 115th decoded
    175
    """

    def __init__(self, pts, dts, key, tb, fps, start=0):
        order = np.lexsort((np.asarray(dts), np.asarray(pts)))  # presentation order, ties by dts
        pts = np.asarray(pts)[order]
        key = np.asarray(key, dtype=bool)[order]
        self.kidx = np.flatnonzero(key)
        self.kpts = pts[self.kidx]
        assert (np.diff(self.kpts) > 0).all(), "keyframe pts not increasing"
        self.n, self.tb, self.fps, self.start = len(pts), tb, fps, start
        self.pts = pts

    def number(self, p):
        """OpenCV ``dts_to_frame_number``: (int64)(fps * sec + 0.5), truncated towards zero."""
        return int(self.fps * (p - self.start) * self.tb + 0.5)

    def frame(self, t):
        n_target = int(math.floor(t * self.fps + 0.5))
        if n_target <= 1:
            return n_target
        delta = 16
        while True:
            temp = max(n_target - delta, 0)
            ts = self.start + int(temp / self.fps / self.tb + 0.5)
            j = max(int(np.searchsorted(self.kpts, ts, side="right")) - 1, 0)
            fn = self.number(self.kpts[j])
            if 0 <= fn <= n_target - 1:
                return min(int(self.kidx[j]) + n_target - fn, self.n - 1)
            if temp == 0:  # OpenCV gives up: the next read is the frame after the keyframe
                return int(self.kidx[j]) + 1
            delta = delta * 2 if delta < 16 else delta * 3 // 2


def probe_video(path):
    """Packet table (pts, dts, keyframe) and stream timing of the video stream, via ffprobe."""
    import subprocess
    from fractions import Fraction

    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base,start_pts",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)["streams"][0]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,dts,flags",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    pts, dts, key = [], [], []
    for line in out.splitlines():
        a, b, fl = line.split(",")[:3]
        pts.append(int(a))
        dts.append(int(b) if b not in ("", "N/A") else int(a))
        key.append(fl.startswith("K"))
    return pts, dts, key, float(Fraction(info["time_base"])), int(info.get("start_pts", 0))


def preprocess(frame_bgr):
    """uint8 [3, 224, 224] exactly as `Video_1fps` (``* 255`` of the stored float tensors)."""
    from removeBlackBorders import center_crop, removeBlackBorders
    from preprocess_frames import scale
    from torchvision.transforms.functional import pil_to_tensor, to_pil_image

    im = to_pil_image(frame_bgr)  # BGR array read as RGB, as the original pipeline did
    im, _ = removeBlackBorders(im)
    im = scale(224, im)
    im = center_crop(im, 224)
    return pil_to_tensor(im).numpy()


def video_seconds(patient):
    """Number of 1 fps rows (seconds) and the CSV frame ids, for the clamp and the fps check."""
    import pandas as pd

    lab = pd.read_csv(FRAME_DIR / patient / "Label" / f"{patient}_all_labels.csv")
    lab = lab.drop_duplicates("second").sort_values("second")
    assert (lab["second"].to_numpy() == np.arange(len(lab))).all(), patient
    return len(lab), lab["frame_id"].to_numpy()


def group_positions(grid, n_seconds, max_group=256, max_span=64 * GRID):
    """
    Split sorted grid positions into groups of at most ``max_group`` positions spanning at
    most ``max_span`` grid steps; each group travels with the frames of its clips.

    >>> [len(g) for g in group_positions(list(range(10)) + [1000, 1001], 500, max_group=4)]
    [4, 4, 2, 2]
    """
    groups, cur = [], []
    for g in grid:
        if cur and (len(cur) >= max_group or g - cur[0] > max_span):
            groups.append(cur)
            cur = []
        cur.append(g)
    if cur:
        groups.append(cur)
    return groups


def decode_worker(job_q, gpu_q, res_q, args):
    """CPU process: decode videos, emit (patient, group id, positions, clip index, frames)."""
    import cv2

    sys.path.insert(0, DATA_DIR)
    cv2.setNumThreads(1)
    while True:
        job = job_q.get()
        if job is None:
            return
        patient, grid = job
        try:
            t0 = time.time()
            n_sec, csv_ids = video_seconds(patient)
            cap = cv2.VideoCapture(str(VIDEO_DIR / patient / f"{patient}.mp4"), cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_N_THREADS, args.decode_threads])
            fps = cap.get(cv2.CAP_PROP_FPS)
            pts, dts, key, tb, start = probe_video(VIDEO_DIR / patient / f"{patient}.mp4")
            em = SeekEmulator(pts, dts, key, tb, fps, start)
            # Video_1fps/{frame_id}.pt was named round(s * fps) with this same fps
            bad = int(np.sum(np.round(np.arange(n_sec) * fps).astype(int) != csv_ids))
            if bad:
                raise RuntimeError(f"{patient}: {bad} CSV frame ids differ from round(s * {fps})")
            groups = group_positions(grid, n_sec)
            need_g = [sorted({f for g in grp for f in clip_grid(g, n_sec)}) for grp in groups]
            if args.check_frames:  # every integer second, compared below, not sent to the GPU
                check_g = set(range(0, n_sec * GRID, GRID))
            else:
                check_g = set()
            wanted = {}  # frame index -> grid indices shown by it
            for gs in need_g:
                for f in gs:
                    wanted.setdefault(em.frame(f / GRID), set()).add(f)
            for f in check_g:
                wanted.setdefault(em.frame(f / GRID), set()).add(f)
            group_last = [max(em.frame(f / GRID) for f in gs) for gs in need_g]
            last_needed = {}  # grid index -> last group using it (to free the buffer)
            for gi, gs in enumerate(need_g):
                for f in gs:
                    last_needed[f] = gi
            buf = {}
            frame_diff, n_checked, n_exact, pts_bad = 0.0, 0, 0, 0
            next_group, idx, last = 0, 0, None
            max_frame = max(wanted) if wanted else -1

            def flush(upto_frame, final=False):
                nonlocal next_group
                while next_group < len(groups):
                    gs = need_g[next_group]
                    if not final and group_last[next_group] > upto_frame:
                        break
                    local = {f: i for i, f in enumerate(gs)}
                    frames = np.stack([buf[f] for f in gs])
                    clip_idx = np.array([[local[f] for f in clip_grid(g, n_sec)] for g in groups[next_group]])
                    gpu_q.put((patient, next_group, np.array(groups[next_group]), clip_idx, frames))
                    for f in gs:
                        if last_needed.get(f) == next_group:
                            buf.pop(f, None)
                    next_group += 1

            while idx <= max_frame:
                if not cap.grab():
                    break
                if idx in wanted:
                    ok, fr = cap.retrieve()
                    assert ok, (patient, idx)
                    # the emulator's presentation order must be the decoder's (frame 0 reports its dts)
                    pts_bad += idx > 0 and round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 / tb) + start != em.pts[idx]
                    x = preprocess(fr)
                    last = x
                    for f in wanted[idx]:
                        if f in check_g:
                            import torch

                            ref = torch.load(
                                FRAME_DIR / patient / "Video_1fps" / f"{int(round(f // GRID * fps)):06d}.pt",
                                weights_only=True,
                            )
                            xt = torch.from_numpy(x) / 255.0
                            frame_diff = max(frame_diff, float((xt - ref).abs().max()))
                            n_exact += bool(torch.equal(xt, ref))
                            n_checked += 1
                        if f in last_needed:
                            buf[f] = x
                    flush(idx)
                idx += 1
            cap.release()
            missing = sum(1 for f in wanted if f >= idx)
            if missing and last is not None:  # video shorter than its last needed frame
                for fi in [f for f in wanted if f >= idx]:
                    for f in wanted[fi]:
                        if f in last_needed:
                            buf[f] = last
            flush(idx, final=True)
            res_q.put(("decoded", patient, {"groups": len(groups), "decoded_frames": idx, "missing_frames": missing,
                                            "frame_max_abs_diff": frame_diff, "frames_checked": n_checked,
                                            "frames_exact": n_exact, "pts_mismatch": int(pts_bad),
                                            "fps": fps, "decode_s": time.time() - t0}))
        except Exception:
            res_q.put(("error", patient, traceback.format_exc()))


def gpu_worker(gpu, gpu_q, res_q, args):
    """GPU process: encoder of ``--run`` on every clip group."""
    import torch

    from infer_untrimmed import load_cfg_and_model

    cfg, model = load_cfg_and_model(Path(args.run) / args.ckpt, gpu)
    assert (cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE, cfg.DATA.SEQ_MODE) == (16, 1, "center"), "clip geometry"
    while True:
        item = gpu_q.get()
        if item is None:
            return
        patient, gid, grid, clip_idx, frames = item
        with torch.no_grad():
            fr = torch.from_numpy(frames).cuda(gpu).float() / 255.0
            idx = torch.from_numpy(clip_idx).cuda(gpu)
            feats = []
            for s in range(0, len(idx), args.batch_size):
                clips = fr[idx[s : s + args.batch_size]].permute(0, 2, 1, 3, 4).contiguous()  # B,C,T,H,W
                preds = model([clips])
                if isinstance(preds, tuple):
                    preds = preds[0]
                cls = preds["cls_tokens"]
                tokens = preds["features"][:, 0 if cls is None else cls.shape[1] :]
                parts = ([] if cls is None else [cls.flatten(1)]) + [tokens.mean(1)]
                feats.append(torch.cat(parts, dim=1).half().cpu())
        res_q.put(("feat", patient, (gid, grid, torch.cat(feats).numpy())))


def load_positions(args, patient, n_sec):
    """Sorted unique grid indices of the positions to extract (clamped to the video)."""
    if args.integer_positions:
        return list(range(0, n_sec * GRID, GRID))
    pos = np.load(Path(args.positions_dir) / f"{patient}.npy")
    g = np.round(pos * GRID)
    assert np.allclose(g / GRID, pos, atol=1e-6), f"{patient}: positions off the 1/{GRID} s grid"
    return sorted(set(np.clip(g.astype(int), 0, (n_sec - 1) * GRID).tolist()))


def compare_features(feat_dir, patient, positions, feats):
    """Cosine between our features at integer positions and the 1 fps ones."""
    ref = np.load(Path(feat_dir) / f"{patient}.npz")
    sec = {int(s): i for i, s in enumerate(ref["second"])}
    keep = [i for i, p in enumerate(positions) if float(p).is_integer() and int(p) in sec]
    a = feats[keep].astype(np.float32)
    b = ref["features"][[sec[int(positions[i])] for i in keep]].astype(np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return {"n": len(keep), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "max_abs_diff": float(np.abs(a - b).max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="encoder run dir (checkpoint + cfg)")
    ap.add_argument("--ckpt", default="checkpoint_best_steps.pyth")
    ap.add_argument("--positions-dir", help="<patient>.npy with the positions (s) of each video")
    ap.add_argument("--integer-positions", action="store_true", help="every integer second (fidelity check)")
    ap.add_argument("--patients", nargs="*", help="default: every .npy of --positions-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--workers", type=int, default=16, help="decoding processes")
    ap.add_argument("--decode-threads", type=int, default=3, help="ffmpeg threads per decoding process")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--check-frames", action="store_true")
    ap.add_argument("--check-features", help="1 fps feature dir of the same encoder")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patients = args.patients or sorted(p.stem for p in Path(args.positions_dir).glob("RARP*.npy"))
    jobs = []
    for p in patients:
        if (out / f"{p}.npz").exists() and not args.overwrite:
            print(f"{p}: exists, skipping", flush=True)
            continue
        n_sec, _ = video_seconds(p)
        jobs.append((p, load_positions(args, p, n_sec)))
    jobs.sort(key=lambda j: -len(j[1]))  # longest first
    print(f"{len(jobs)} videos, {sum(len(j[1]) for j in jobs)} positions", flush=True)

    ctx = mp.get_context("spawn")
    job_q, gpu_q, res_q = ctx.Queue(), ctx.Queue(maxsize=8 * len(args.gpus)), ctx.Queue()
    gpus = [ctx.Process(target=gpu_worker, args=(g, gpu_q, res_q, args), daemon=True) for g in args.gpus]
    decs = [ctx.Process(target=decode_worker, args=(job_q, gpu_q, res_q, args), daemon=True)
            for _ in range(min(args.workers, len(jobs)))]
    for pr in gpus + decs:
        pr.start()
    for j in jobs:
        job_q.put(j)
    for _ in decs:
        job_q.put(None)

    t0 = time.time()
    pending = {p: {"n_groups": None, "parts": {}} for p, _ in jobs}
    fidelity = json.loads((out / "fidelity.json").read_text()) if (out / "fidelity.json").exists() else {}
    stats = {}
    done = 0
    while done < len(jobs):
        kind, patient, payload = res_q.get()
        if kind == "error":
            raise RuntimeError(f"{patient} failed:\n{payload}")
        st = pending[patient]
        if kind == "decoded":
            st["n_groups"] = payload["groups"]
            stats[patient] = payload
        else:
            gid, grid, feats = payload
            st["parts"][gid] = (grid, feats)
        if st["n_groups"] is not None and len(st["parts"]) == st["n_groups"]:
            grid = np.concatenate([st["parts"][i][0] for i in range(st["n_groups"])])
            feats = np.concatenate([st["parts"][i][1] for i in range(st["n_groups"])])
            positions = (grid / GRID).astype(np.float32)
            tmp = out / f"{patient}.tmp.npz"
            np.savez(tmp, positions=positions, features=feats)
            tmp.rename(out / f"{patient}.npz")
            info = dict(stats[patient], positions=len(positions))
            if args.check_frames or args.check_features:
                if args.check_features:
                    info["features"] = compare_features(args.check_features, patient, positions, feats)
                fidelity[patient] = info
                (out / "fidelity.json").write_text(json.dumps(fidelity, indent=1))
            done += 1
            del pending[patient]
            print(f"[{time.time() - t0:.0f}s] {patient} ({done}/{len(jobs)}): {json.dumps(info)}", flush=True)
            free_gb = shutil.disk_usage(out).free / 1e9
            if free_gb < MIN_FREE_GB:
                raise RuntimeError(f"only {free_gb:.0f} GB free on {out}: stopping")
    for _ in gpus:
        gpu_q.put(None)
    for pr in gpus + decs:
        pr.join(timeout=60)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
