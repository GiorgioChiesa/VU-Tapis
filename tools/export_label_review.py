#!/usr/bin/env python3
"""
Data of the label-review web app (``tools/label_review.html``, served by
``tools/serve_viewer.py --api``): one item per training keyframe, with its 64 frames at 4 fps.

Items are the rows the frame-level training used, rebuilt with the training code itself
(``eval_untrimmed.build_label_table`` -> ``Orsi._load_data``: ``EXCLUDE_EVENT_NAMES``,
``_add_idle_frames`` with ``ADD_IDLE`` 10 and ``random_state=1``, ``collapse_event_dfs``):

  * events:   every non-Idle row of the 50 ``RARPxx_all_labels.csv`` kept by training;
  * controls: the Idle rows ``_add_idle_frames`` picks for the patient. The pick is per patient
    (it only looks at that patient's rows), so it is the same whichever split holds the patient.
    Its "30 s from any event" test compares ``frame_id`` (~60 per second) with 30, so the real
    exclusion zone is only about +-0.5 s. Kept as is: these are the rows training saw.

Each item carries the out-of-fold prediction of the best method
(``outputs/orsi/TEMPORAL/cv_ft_dilate3/fold*/val/<p>.npz``, ``probs_steps`` of the CSV row of that
second); class order is the fold's training order, rebuilt as ``eval_constraints.prepare``.

Frames: times ``second - 8 ... second + 8 - 0.25`` (0.25 s steps, clamped to the video), keyframe
at index 32. Each frame is the one OpenCV returns after ``cap.set(CAP_PROP_POS_MSEC, t * 1000)``,
cropped as ``Video_1fps`` (``extract_shifted_features.SeekEmulator`` / ``preprocess``), so frame 32
is the stored ``.pt`` of the second. Saved as RGB 224 jpg (the tensors are BGR).

Output (``--out``, default ``/home/gchie/workspace/viewer/revisione_etichette``):
  items.json                          classes, key map, items
  frames/<p>_t<sec:05d>/NN.jpg        NN = 00..63
  frames/.done/<p>                    one marker per finished video (the run is resumable)
  export_summary.json                 counts, timings, keyframe check (aggregates)

CPU only, one worker process per video. Example (Vu-Tapis env, from src/VU-Tapis):
    python tools/export_label_review.py --workers 12
    python tools/export_label_review.py --patients RARP07 --check 20     # pilot
"""

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "detectron2", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

MAIN_REPO = Path("/home/gchie/workspace/thesis_GiorgioChiesa/src/VU-Tapis")
TEMPORAL_ROOT = MAIN_REPO / "outputs/orsi/TEMPORAL/cv_ft_dilate3"
FRAME_DIR = Path("/home/gchie/workspace/data/orsi_tensors")
VIDEO_DIR = Path("/home/gchie/workspace/nas_private/data/orsi/videos")
OUT_DIR = Path("/home/gchie/workspace/viewer/revisione_etichette")

FPS = 4  # review frames per second
HALF = 8  # seconds before / after the keyframe (16 s clip of the model)
N_FRAMES = 2 * HALF * FPS  # 64
KEY_INDEX = HALF * FPS  # 32
# keys that assign a class, in this order: QWERTY letters, then digits
KEYS = list("qwertyuiopasdfghjklzxcvbnm") + list("1234567")


def key_map(names):
    """Class index -> key, in class order (33 classes: 26 letters, then 1..7)."""
    if len(names) > len(KEYS):
        raise ValueError(f"{len(names)} classes but only {len(KEYS)} keys")
    return {i: KEYS[i] for i in range(len(names))}


def window_times(second, t_max):
    """
    The 64 frame times (s) of a keyframe: ``second - 8 + i / 4``, clamped to ``[0, t_max]``.

    >>> w = window_times(100, 3600.0); len(w), w[0], w[32], w[-1]
    (64, 92.0, 100.0, 107.75)
    >>> window_times(3, 3600.0)[19:22], window_times(3590, 3595.5)[-1]
    ([0.0, 0.0, 0.25], 3595.5)
    """
    return [min(max(second - HALF + i / FPS, 0.0), t_max) for i in range(N_FRAMES)]


def frames_dir_name(patient, second):
    return f"{patient}_t{int(second):05d}"


# ---------------------------------------------------------------------------------- items
def fold_of_patients(temporal_root):
    folds = {}
    for d in sorted(Path(temporal_root).glob("fold*/val")):
        meta = json.loads((d / "meta.json").read_text())
        for p in meta["patients"]:
            folds[p] = (int(d.parent.name.removeprefix("fold")), d)
    return folds


def build_items(temporal_root, frame_dir, work_dir, patients=None):
    """Items (events + controls) of every patient, with the out-of-fold prediction."""
    import pandas as pd

    from eval_untrimmed import build_label_table, global_to_local, load_cfg

    folds = fold_of_patients(temporal_root)
    items, names_ref, ph_ref, l2g_ref = [], None, None, None
    raw_names = {}
    for fold in sorted({f for f, _ in folds.values()}):
        pred_dir = next(d for f, d in folds.values() if f == fold)
        meta = json.loads((pred_dir / "meta.json").read_text())
        run = Path(meta["run"])
        cfg = load_cfg(run, meta["ckpt"])
        train = meta.get("train_patients") or [Path(f).name.removesuffix(".csv") for f in cfg.ENDOVIS_DATASET.TRAIN_LISTS]
        work = Path(work_dir) / f"fold{fold}"
        work.mkdir(parents=True, exist_ok=True)
        # class order exactly as eval_constraints.prepare (training patients of the fold)
        _, tr_ids = build_label_table(cfg, train, str(frame_dir), work)
        g2l, names = global_to_local(run, "steps", tr_ids["steps"])
        g2l_ph, ph_names = global_to_local(run, "phases", tr_ids["phases"])
        if names_ref is None:
            names_ref, ph_ref = names, ph_names
            l2g_ref = {v: int(k) for k, v in g2l.items()}
        elif names != names_ref or ph_names != ph_ref:
            raise RuntimeError(f"fold {fold}: class order differs from fold 0")
        val = [p for p in meta["patients"] if patients is None or p in patients]
        if not val:
            continue
        table, _ = build_label_table(cfg, val, str(frame_dir), work)
        idle = [i for i, n in enumerate(names) if n.lower() == "idle"]
        assert len(idle) == 1
        for p in val:
            raw = pd.read_csv(Path(frame_dir) / p / "Label" / f"{p}_all_labels.csv")
            for gid, nm in zip(raw["event_id"], raw["event_name"]):
                raw_names.setdefault(int(gid), nm)
            npz = np.load(pred_dir / f"{p}.npz")
            sec_row = {}
            for r, s in enumerate(npz["second"].tolist()):
                sec_row.setdefault(int(s), r)
            probs = npz["probs_steps"]
            rows = table[table["patient_name"] == p]
            for _, row in rows.iterrows():
                s = int(row["second"])
                t = g2l[int(row["event_id"])]
                pr = probs[sec_row[s]].astype(float)
                top = np.argsort(-pr)[:3]
                pred = int(top[0])
                is_event = t != idle[0]
                items.append(dict(
                    id=f"{p}_t{s:05d}_{'e' if is_event else 'c'}{t:02d}",
                    patient=p, fold=fold, second=s,
                    type="event" if is_event else "control",
                    true=t, phase=g2l_ph.get(int(row["phase_id"]), -1),
                    csv_event_name=str(row["event_name"]),
                    pred=pred, conf=round(float(pr[pred]), 4), p_true=round(float(pr[t]), 4),
                    top3=[[int(c), round(float(pr[c]), 4)] for c in top],
                    correct=bool(pred == t),
                    frames=frames_dir_name(p, s),
                ))
    items.sort(key=lambda it: (it["patient"], it["second"], it["id"]))
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return dict(
        names=names_ref, phase_names=ph_ref,
        class_global_ids=[l2g_ref[i] for i in range(len(names_ref))],
        class_csv_names=[raw_names.get(l2g_ref[i], names_ref[i]) for i in range(len(names_ref))],
        keys=[key_map(names_ref)[i] for i in range(len(names_ref))],
        fps=FPS, n_frames=N_FRAMES, key_index=KEY_INDEX,
        items=items,
    )


# ---------------------------------------------------------------------------------- frames
def find_video(p):
    from add_tool_overlay import find_video as _find

    return _find(VIDEO_DIR, p)


def to_rgb_hwc(x):
    """uint8 [3, H, W] in the tensors' BGR order -> RGB [H, W, 3]."""
    return np.ascontiguousarray(x.transpose(1, 2, 0)[..., ::-1])


def segments(windows, gap):
    """
    Merge keyframe windows ``[(lo_frame, hi_frame, key), ...]`` whose frame ranges are less than
    ``gap`` frames apart, so each segment needs one seek and one forward decode.

    >>> segments([(0, 10, 'a'), (5, 20, 'b'), (100, 120, 'c')], gap=50)
    [(0, 20, ['a', 'b']), (100, 120, ['c'])]
    """
    out = []
    for lo, hi, key in sorted(windows):
        if out and lo <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], hi), out[-1][2] + [key])
        else:
            out.append((lo, hi, [key]))
    return out


def video_worker(p, seconds, check_seconds, out_dir, quality):
    """Write the 64 jpgs of every keyframe second of video ``p``; return timing and checks."""
    import cv2
    import torch
    from PIL import Image

    import extract_shifted_features as esf

    sys.path.insert(0, esf.DATA_DIR)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    t0 = time.time()
    path = find_video(p)
    n_sec, csv_ids = esf.video_seconds(p)
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    fps = cap.get(cv2.CAP_PROP_FPS)
    pts, dts, key, tb, start = esf.probe_video(path)
    em = esf.SeekEmulator(pts, dts, key, tb, fps, start)
    t_max = math.floor((int(em.pts[-1]) - start) * tb * FPS) / FPS
    frame_idx = {}  # second -> 64 decoded-frame indices
    for s in seconds:
        frame_idx[s] = [em.frame(t) for t in window_times(s, t_max)]
    segs = segments([(min(v), max(v), s) for s, v in frame_idx.items()], gap=int(4 * fps))
    root = Path(out_dir) / "frames"
    checks, pts_bad, n_written, n_dup = [], 0, 0, 0
    for lo, hi, secs in segs:
        wanted = {f for s in secs for f in frame_idx[s]}
        # the frame returned after seeking to the time of `lo` is `lo` (SeekEmulator)
        t_lo = min(t for s in secs for t, f in zip(window_times(s, t_max), frame_idx[s]) if f == lo)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_lo * 1000)
        buf, idx, last = {}, lo, None
        while idx <= hi:
            if not cap.grab():
                break
            if idx in wanted:
                ok, fr = cap.retrieve()
                assert ok, (p, idx)
                pts_bad += idx > 0 and round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 / tb) + start != em.pts[idx]
                buf[idx] = last = esf.preprocess(fr)
            idx += 1
        for f in wanted - set(buf):  # past the last decodable frame
            buf[f] = last
        for s in secs:
            d = root / frames_dir_name(p, s)
            d.mkdir(parents=True, exist_ok=True)
            n_dup += len(frame_idx[s]) - len(set(frame_idx[s]))
            for i, f in enumerate(frame_idx[s]):
                Image.fromarray(to_rgb_hwc(buf[f])).save(d / f"{i:02d}.jpg", quality=quality)
                n_written += 1
            if s in check_seconds:
                ref = torch.load(FRAME_DIR / p / "Video_1fps" / f"{int(csv_ids[s]):06d}.pt", weights_only=True)
                ref = (ref * 255).round().to(torch.uint8).numpy()  # BGR [3, 224, 224]
                ref_rgb = to_rgb_hwc(ref).astype(np.int16)
                raw = to_rgb_hwc(buf[frame_idx[s][KEY_INDEX]]).astype(np.int16)
                jpg = np.asarray(Image.open(d / f"{KEY_INDEX:02d}.jpg").convert("RGB")).astype(np.int16)
                checks.append(dict(
                    raw_max=int(np.abs(raw - ref_rgb).max()), raw_mean=float(np.abs(raw - ref_rgb).mean()),
                    jpg_max=int(np.abs(jpg - ref_rgb).max()), jpg_mean=float(np.abs(jpg - ref_rgb).mean()),
                    # same comparison without the BGR->RGB flip of the stored tensor: must be worse
                    jpg_mean_noflip=float(np.abs(jpg - ref.transpose(1, 2, 0).astype(np.int16)).mean()),
                ))
    cap.release()
    return dict(patient=p, keyframes=len(seconds), segments=len(segs), jpgs=n_written,
                clamped_or_repeated_frames=n_dup, pts_mismatch=int(pts_bad), checks=checks,
                seconds_elapsed=round(time.time() - t0, 1))


def run_video(p, seconds, check_seconds, out_dir, quality):
    try:
        res = video_worker(p, seconds, check_seconds, out_dir, quality)
        done = Path(out_dir) / "frames" / ".done"
        done.mkdir(parents=True, exist_ok=True)
        (done / p).write_text(json.dumps({k: v for k, v in res.items() if k != "checks"}))
        return res
    except Exception:
        return dict(patient=p, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temporal-root", default=str(TEMPORAL_ROOT))
    ap.add_argument("--frame-dir", default=str(FRAME_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--patients", nargs="*", help="extract frames only for these (items.json always has all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", type=int, default=20, help="random keyframes compared with their .pt")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--items-only", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items_path = out / "items.json"
    if items_path.exists():
        data = json.loads(items_path.read_text())
    else:
        t = time.time()
        data = build_items(args.temporal_root, args.frame_dir, out / "_work")
        tmp = items_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(items_path)
        print(f"items.json: {len(data['items'])} items in {time.time() - t:.0f} s", flush=True)
    if args.items_only:
        return

    by_p = {}
    for it in data["items"]:
        by_p.setdefault(it["patient"], set()).add(it["second"])
    todo = [p for p in sorted(by_p) if (args.patients is None or p in args.patients)
            and not (out / "frames" / ".done" / p).exists()]
    rng = random.Random(0)
    pool_keys = [(p, s) for p in todo for s in sorted(by_p[p])]
    check = set(rng.sample(pool_keys, min(args.check, len(pool_keys))))
    print(f"{len(todo)} videos to extract, {len(pool_keys)} keyframes, {len(check)} checked", flush=True)
    t0, results = time.time(), []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_video, p, sorted(by_p[p]), {s for q, s in check if q == p}, str(out), args.quality)
                for p in todo]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            if "error" in r:
                print(f"ERROR {r['patient']}\n{r['error']}", flush=True)
            else:
                print(f"{r['patient']}: {r['keyframes']} keyframes, {r['segments']} segments, "
                      f"{r['seconds_elapsed']} s, pts mismatch {r['pts_mismatch']} "
                      f"[{len(results)}/{len(todo)}, {time.time() - t0:.0f} s]", flush=True)
    checks = [c for r in results for c in r.get("checks", [])]
    summ = dict(
        videos=len(results), errors=[r["patient"] for r in results if "error" in r],
        wall_s=round(time.time() - t0, 1),
        video_s=[r.get("seconds_elapsed") for r in results],
        keyframes=sum(r.get("keyframes", 0) for r in results),
        jpgs=sum(r.get("jpgs", 0) for r in results),
        pts_mismatch=sum(r.get("pts_mismatch", 0) for r in results),
        check=dict(n=len(checks), **{f"{k}_{agg}": (fn([c[k] for c in checks]) if checks else None)
                                     for k in ("raw_max", "raw_mean", "jpg_max", "jpg_mean", "jpg_mean_noflip")
                                     for agg, fn in (("max", max), ("mean", lambda v: float(np.mean(v))))}),
    )
    name = "export_summary.json" if args.patients is None else f"export_summary_{'_'.join(args.patients)}.json"
    (out / name).write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "video_s"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
