#!/usr/bin/env python3
"""
4 fps frame cache of the Orsi videos, built exactly like `Video_1fps` but on a 0.25 s grid.

Row ``k`` of a video is the frame OpenCV returns after ``cap.set(CAP_PROP_POS_MSEC, k * 250)``
(emulated from the packet table by `extract_shifted_features.SeekEmulator`, so each video is decoded
once, sequentially), preprocessed as `data/check_webapp/main.py:process_video` did (BGR array read as
RGB, per-frame black-border crop, short side 224, centre crop 224) and stored as uint8.

Rows cover ``k = 0 .. 4 * n_sec - 1``, where ``n_sec`` is the number of 1 fps rows of the original
label CSV, so that row ``4 * s`` is the frame of second ``s`` (= ``Video_1fps`` of that second).
Grid times past the last decodable frame repeat the last decoded frame (counted in ``missing``).

Output per video (``<out>/<patient>/``):
  * ``frames_4fps.npy``: uint8 [4 * n_sec, 3, 224, 224] (load with ``np.load(..., mmap_mode="r")``);
  * ``meta.json``: aggregate counts and the integer-second fidelity check;
  * ``.done``: written last; videos with it are skipped (resumable).

``--check-every S`` compares every S-th integer-second row with its ``Video_1fps/*.pt`` (S = 1:
every second). Expected max abs diff 0.

Example:
    python tools/build_4fps_frames.py --out /home/gchie/workspace/data/orsi_tensors_4fps --workers 20
"""

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from extract_shifted_features import DATA_DIR, FRAME_DIR, VIDEO_DIR, SeekEmulator, preprocess, probe_video, video_seconds

GRID = 4
MIN_FREE_GB = 100


def build_video(patient, args):
    import cv2
    import torch

    sys.path.insert(0, DATA_DIR)
    cv2.setNumThreads(1)
    out_dir = Path(args.out) / patient
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_sec, csv_ids = video_seconds(patient)
    n_rows = n_sec * GRID
    video = VIDEO_DIR / patient / f"{patient}.mp4"
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, args.decode_threads])
    fps = cap.get(cv2.CAP_PROP_FPS)
    bad = int(np.sum(np.round(np.arange(n_sec) * fps).astype(int) != csv_ids))
    if bad:
        raise RuntimeError(f"{patient}: {bad} CSV frame ids differ from round(s * {fps})")
    pts, dts, key, tb, start = probe_video(video)
    em = SeekEmulator(pts, dts, key, tb, fps, start)
    wanted = {}  # decoded frame index -> grid rows
    for k in range(n_rows):
        wanted.setdefault(em.frame(k / GRID), []).append(k)
    check = set(range(0, n_rows, GRID * args.check_every)) if args.check_every > 0 else set()

    tmp = out_dir / "frames_4fps.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_rows, 3, 224, 224))
    filled = np.zeros(n_rows, dtype=bool)
    max_frame = max(wanted)
    diff, n_checked, n_exact, idx, last = 0.0, 0, 0, 0, None
    while idx <= max_frame:
        if not cap.grab():
            break
        if idx in wanted:
            ok, fr = cap.retrieve()
            assert ok, (patient, idx)
            x = preprocess(fr)
            last = x
            for k in wanted[idx]:
                arr[k] = x
                filled[k] = True
                if k in check:
                    ref = torch.load(FRAME_DIR / patient / "Video_1fps" / f"{int(csv_ids[k // GRID]):06d}.pt",
                                     weights_only=True)
                    xt = torch.from_numpy(x) / 255.0
                    diff = max(diff, float((xt - ref).abs().max()))
                    n_exact += bool(torch.equal(xt, ref))
                    n_checked += 1
        idx += 1
    cap.release()
    missing = int((~filled).sum())
    if missing:
        if last is None:
            raise RuntimeError(f"{patient}: nothing decoded")
        first_missing = int(np.flatnonzero(~filled)[0])
        if filled[first_missing:].any():
            raise RuntimeError(f"{patient}: unfilled rows inside the video")
        arr[first_missing:] = last
    arr.flush()
    del arr
    tmp.rename(out_dir / "frames_4fps.npy")
    meta = {"patient": patient, "n_sec": n_sec, "n_rows": n_rows, "fps": fps, "decoded_frames": idx,
            "missing_rows": missing, "check_every_s": args.check_every, "frames_checked": n_checked,
            "frames_exact": n_exact, "frame_max_abs_diff": diff, "build_s": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / ".done").write_text("")
    return meta


def worker(job_q, res_q, args):
    while True:
        patient = job_q.get()
        if patient is None:
            return
        try:
            res_q.put(("ok", patient, build_video(patient, args)))
        except Exception:
            res_q.put(("error", patient, traceback.format_exc()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", nargs="*", help="default: every patient of the 1 fps tensors")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--decode-threads", type=int, default=3)
    ap.add_argument("--check-every", type=int, default=10, help="compare every S-th second with Video_1fps (0: off)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patients = args.patients or sorted(p.name for p in FRAME_DIR.glob("RARP*") if (p / "Label").is_dir())
    todo = [p for p in patients if not (out / p / ".done").exists()]
    todo.sort(key=lambda p: -video_seconds(p)[0])  # longest first
    print(f"{len(patients)} videos, {len(todo)} to build", flush=True)
    ctx = mp.get_context("spawn")
    job_q, res_q = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(job_q, res_q, args), daemon=True)
             for _ in range(min(args.workers, len(todo)))]
    for pr in procs:
        pr.start()
    for p in todo:
        job_q.put(p)
    for _ in procs:
        job_q.put(None)
    t0, errors = time.time(), []
    for i in range(len(todo)):
        kind, patient, payload = res_q.get()
        if kind == "error":
            errors.append(patient)
            print(f"{patient} FAILED:\n{payload}", flush=True)
            continue
        print(f"[{time.time() - t0:.0f}s] {patient} ({i + 1}/{len(todo)}): {json.dumps(payload)}", flush=True)
        free_gb = shutil.disk_usage(out).free / 1e9
        if free_gb < MIN_FREE_GB:
            raise RuntimeError(f"only {free_gb:.0f} GB free on {out}: stopping")
    for pr in procs:
        pr.join(timeout=60)
    print(f"done in {time.time() - t0:.0f}s, errors: {errors}", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
