#!/usr/bin/env python3
"""
Decode the original Orsi videos and save, for every row of `Label/{patient}_all_labels.csv`,
the frame with index `frame_id` (the same frame as `Video_1fps/{frame_id:06d}.pt`) as a jpg
with long side `--max-side`. The NAS jpgs (orsi/images) are sampled at integer fps and drift
away from `frame_id` on videos at 58.94 fps, so they cannot be used.

    /home/gchie/miniforge3/envs/vlm/bin/python tools/vlm/extract_frames.py --patients RARP49 RARP50
    -> /home/gchie/workspace/data/orsi_frames448/{patient}/{frame_id:06d}.jpg  (stays on this machine)
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import pandas as pd

VAL = ["RARP49", "RARP50", "RARP59", "RARP61", "RARP62", "RARP64"]


def extract(patient, video_dir, label_dir, out_root, max_side):
    out = Path(out_root) / patient
    out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(Path(label_dir) / patient / "Label" / f"{patient}_all_labels.csv")["frame_id"].astype(int)
    wanted = set(ids) - {int(p.stem) for p in out.glob("*.jpg")}
    if not wanted:
        return f"{patient}: complete ({len(ids)} frames)"
    t0, last = time.time(), max(wanted)
    with av.open(str(Path(video_dir) / patient / f"{patient}.mp4")) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i in wanted:
                im = frame.to_image()
                r = max_side / max(im.size)
                im.resize((round(im.width * r), round(im.height * r))).save(out / f"{i:06d}.jpg", quality=92)
                wanted.discard(i)
            if i >= last:
                break
    return f"{patient}: {len(ids) - len(wanted)}/{len(ids)} frames in {time.time() - t0:.0f}s, missing {len(wanted)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="*", default=VAL)
    ap.add_argument("--video-dir", default="/home/gchie/workspace/nas_private/data/orsi/videos")
    ap.add_argument("--label-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--out", default="/home/gchie/workspace/data/orsi_frames448")
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(extract, p, args.video_dir, args.label_dir, args.out, args.max_side) for p in args.patients]
        for f in futs:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
