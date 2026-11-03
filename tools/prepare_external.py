#!/usr/bin/env python3
"""
Per-second labels (and, for SAR-RARP50, 1 fps frames) of the external RARP datasets
used by Experiment 2: GraSP, SAR-RARP50 and ESAD.

Writes ``<out>/<ds>/<video>/labels.npz`` for every video plus ``<out>/<ds>/meta.json``
(vocabularies, splits, per-video frame dir, excluded videos, aggregate stats).
Every ``labels.npz`` holds:

- ``second``      int32 [T]   time of the row in seconds (sorted, may have gaps for ESAD)
- ``frame_id``    int64 [T]   frame number in the source video/annotation
- ``frame_path``  str   [T]   absolute path of the 1 fps jpg of that second

and, per dataset:

- GraSP: ``phases`` [T] (11), ``steps`` [T] (21), ``actions`` [T,14] and ``instruments``
  [T,7] multi-hot (OR over the instances of the short-term keyframe), ``actions_mask`` [T]
  (1 only at keyframes; elsewhere actions/instruments are unknown, not negative).
- SAR-RARP50: ``gestures`` [T] (8, -1 = no 10 Hz sample in that second) = mode of the
  ``action_discrete.txt`` samples with ``60*s <= frame < 60*(s+1)``.
- ESAD: ``actions`` [T,21] multi-hot from the YOLO txt (empty txt = all negative),
  ``n_instances`` [T].

Only aggregate statistics are printed. Example:
    python tools/prepare_external.py --datasets grasp sar esad --workers 6
"""

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WS = Path("/home/gchie/workspace")
GRASP_DIR = WS / "data/GraSP_1fps"
SAR_DIR = WS / "nas_private/data/SAR-RARP50"
ESAD_DIR = WS / "nas_private/data/ESAD"
OUT_DIR = WS / "data/external_1fps"
FFMPEG = Path(sys.executable).parent / "ffmpeg"
FFPROBE = Path(sys.executable).parent / "ffprobe"

SAR_FPS = 60
SAR_GESTURES = [
    "Other",
    "Picking-up the needle",
    "Positioning the needle tip",
    "Pushing the needle through the tissue",
    "Pulling the needle out of the tissue",
    "Tying a knot",
    "Cutting the suture",
    "Returning/dropping the needle",
]
# ESAD procedures are named RARP1..4 by the challenge; prefixed to never clash with Orsi ids.
ESAD_SPLITS = {
    "train": [("ESAD_RARP2", "train/train/set1", "train/train/set1"), ("ESAD_RARP4", "train/train/set2", "train/train/set2")],
    "val": [("ESAD_RARP1", "val/val/obj", "val/val/obj")],
    "test": [("ESAD_RARP3", "test_labels/test_labels", "test_images/test")],
}

EXPECTED = {
    "grasp": {"seconds": {"train": 73619, "test": 42897}, "keyframes": {"train": 2324, "test": 1125}},
    "sar": {"rows_10hz": 162705},
    "esad": {"instances": {"train": 27998, "val": 7120, "test": 11207}},
}


# --------------------------------------------------------------------------- utils


def save_video(out_dir, video, arrays):
    d = out_dir / video
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "labels.npz", **arrays)


def multi_hot(lists, n, offset):
    """[[ids...], ...] -> uint8 [len(lists), n] with column ``id - offset``."""
    out = np.zeros((len(lists), n), dtype=np.uint8)
    for i, ids in enumerate(lists):
        for c in ids:
            out[i, c - offset] = 1
    return out


def gesture_per_second(frames, labels, n_seconds, fps=SAR_FPS, n_classes=len(SAR_GESTURES)):
    """
    Mode of the samples ``(frame, label)`` falling in each second ``[fps*s, fps*(s+1))``;
    ties go to the smallest id, seconds without samples get -1.
    """
    frames = np.asarray(frames, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    sec = frames // fps
    keep = sec < n_seconds
    counts = np.zeros((n_seconds, n_classes), dtype=np.int64)
    np.add.at(counts, (sec[keep], labels[keep]), 1)
    out = counts.argmax(1).astype(np.int16)
    out[counts.sum(1) == 0] = -1
    return out


def class_counts(names, column_sums):
    return {n: int(c) for n, c in zip(names, column_sums)}


def check(name, got, expected):
    ok = got == expected
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: {got} (expected {expected})", flush=True)
    return ok


# --------------------------------------------------------------------------- GraSP


def prepare_grasp(out_root, args):
    out_dir = out_root / "grasp"
    ann = GRASP_DIR / "annotations"
    meta = {"dataset": "grasp", "source": str(GRASP_DIR), "splits": {}, "videos": {}, "excluded": {}, "stats": {}}
    vocab = None
    ok = True
    for split in ("train", "test"):
        long = json.load(open(ann / f"grasp_long-term_{split}.json"))
        short = json.load(open(ann / f"grasp_short-term_{split}.json"))
        if vocab is None:
            vocab = {
                "phases": [c["name"] for c in sorted(long["phases_categories"], key=lambda c: c["id"])],
                "steps": [c["name"] for c in sorted(long["steps_categories"], key=lambda c: c["id"])],
                "actions": [c["name"] for c in sorted(short["actions_categories"], key=lambda c: c["id"])],
                "instruments": [c["name"] for c in sorted(short["categories"], key=lambda c: c["id"])],
            }
            # phases/steps are 0-based (0 = Idle), actions/instruments 1-based.
            assert [c["id"] for c in long["phases_categories"]] == list(range(11))
            assert [c["id"] for c in long["steps_categories"]] == list(range(21))
            assert sorted(c["id"] for c in short["actions_categories"]) == list(range(1, 15))
            assert sorted(c["id"] for c in short["categories"]) == list(range(1, 8))

        img = {im["id"]: im for im in long["images"]}
        rows = collections.defaultdict(dict)  # video -> frame_num -> (phase, step)
        for a in long["annotations"]:
            im = img[a["image_id"]]
            rows[im["video_name"]][im["frame_num"]] = (a["phases"], a["steps"])
        n_long = sum(len(r) for r in rows.values())

        kimg = {im["id"]: im for im in short["images"]}
        kf = collections.defaultdict(lambda: collections.defaultdict(lambda: (set(), set())))
        for im in short["images"]:
            kf[im["video_name"]][im["frame_num"]]  # keyframes without instances stay negatives
        n_inst = 0
        for a in short["annotations"]:
            im = kimg[a["image_id"]]
            acts, insts = kf[im["video_name"]][im["frame_num"]]
            acts.update(a["actions"])
            insts.add(a["instruments"])
            n_inst += 1
        n_kf = sum(len(v) for v in kf.values())

        videos = sorted(rows)
        meta["splits"][split] = videos
        st = {"annotated_seconds": n_long, "seconds": 0, "missing_frames": 0, "keyframes": n_kf,
              "keyframes_kept": 0, "instances": n_inst}
        tot = {k: np.zeros(len(vocab[k]), dtype=np.int64) for k in vocab}
        for v in videos:
            frame_dir = GRASP_DIR / "frames" / v
            secs = np.array(sorted(rows[v]), dtype=np.int32)
            exists = np.array([(frame_dir / f"{s:05d}.jpg").exists() for s in secs])
            st["missing_frames"] += int((~exists).sum())
            secs = secs[exists]
            ph = np.array([rows[v][s][0] for s in secs], dtype=np.int16)
            sp = np.array([rows[v][s][1] for s in secs], dtype=np.int16)
            mask = np.array([s in kf[v] for s in secs], dtype=np.uint8)
            acts = multi_hot([sorted(kf[v][s][0]) if s in kf[v] else [] for s in secs], 14, 1)
            insts = multi_hot([sorted(kf[v][s][1]) if s in kf[v] else [] for s in secs], 7, 1)
            missing_kf = set(kf[v]) - set(secs.tolist())
            assert not missing_kf - set(rows[v]), f"{v}: keyframes outside the long-term annotation"
            save_video(out_dir, v, {
                "second": secs,
                "frame_id": secs.astype(np.int64),
                "frame_path": np.array([str(frame_dir / f"{s:05d}.jpg") for s in secs]),
                "phases": ph,
                "steps": sp,
                "actions": acts,
                "instruments": insts,
                "actions_mask": mask,
            })
            meta["videos"][v] = {"split": split, "n_seconds": int(len(secs)), "n_keyframes": int(mask.sum())}
            st["seconds"] += len(secs)
            st["keyframes_kept"] += int(mask.sum())
            tot["phases"] += np.bincount(ph, minlength=11)
            tot["steps"] += np.bincount(sp, minlength=21)
            tot["actions"] += acts.sum(0, dtype=np.int64)
            tot["instruments"] += insts.sum(0, dtype=np.int64)
        st["per_class_seconds"] = {k: class_counts(vocab[k], tot[k]) for k in ("phases", "steps")}
        st["per_class_keyframes"] = {k: class_counts(vocab[k], tot[k]) for k in ("actions", "instruments")}
        meta["stats"][split] = st
        print(f"GraSP {split}: {len(videos)} videos, {st['seconds']} s with frames "
              f"({st['missing_frames']} annotated s without jpg), {n_kf} keyframes, {n_inst} instances")
        ok &= check(f"grasp {split} annotated seconds", n_long, EXPECTED["grasp"]["seconds"][split])
        ok &= check(f"grasp {split} keyframes", n_kf, EXPECTED["grasp"]["keyframes"][split])
        for k in vocab:
            print(f"    {k}: {st['per_class_seconds' if k in ('phases', 'steps') else 'per_class_keyframes'][k]}")
    meta["vocab"] = vocab
    meta["checks_ok"] = bool(ok)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return ok


# --------------------------------------------------------------------------- SAR-RARP50


def read_discrete(path):
    a = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    return a[:, 0], a[:, 1]


def probe_frames(avi):
    r = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,r_frame_rate",
         "-of", "json", str(avi)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(r.stdout)["streams"][0]
    assert s["r_frame_rate"] == f"{SAR_FPS}/1", (avi, s)
    return int(s["nb_frames"])


def extract_frames(avi, frame_dir, threads=4):
    """Every 60th frame (0, 60, 120, ...) -> frame_dir/NNNNN.jpg with NNNNN = second. Returns (n, n_errors)."""
    tmp = frame_dir.with_name(frame_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    r = subprocess.run(
        [str(FFMPEG), "-nostdin", "-v", "error", "-threads", str(threads), "-i", str(avi),
         "-vf", f"select=not(mod(n\\,{SAR_FPS}))", "-fps_mode", "passthrough", "-q:v", "2",
         "-start_number", "0", str(tmp / "%05d.jpg")],
        capture_output=True, text=True,
    )
    errors = [l for l in r.stderr.splitlines() if l.strip()]
    n = len(list(tmp.glob("*.jpg")))
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    tmp.rename(frame_dir)
    return n, len(errors) + (r.returncode != 0)


def unzip_avi(zip_path, dest):
    """Re-extract video_left.avi from the original zip (CRC verified by zipfile). Returns error str or None."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z, z.open("video_left.avi") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, 16 << 20)
    except (zipfile.BadZipFile, zlib.error, OSError, KeyError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def sar_video(item, out_dir, raw_dir, args):
    split, folder = item
    name = folder.name.removesuffix(".tmp")
    info = {"split": split, "source": str(folder), "corrupt_folder": folder.name.endswith(".tmp")}
    frames, labels = read_discrete(folder / "action_discrete.txt")
    info["rows_10hz"] = int(len(frames))
    avi = folder / "video_left.avi"
    if info["corrupt_folder"]:
        zip_path = folder.parent / "zips" / f"{name}.zip"
        avi = raw_dir / name / "video_left.avi"
        if not (raw_dir / name / ".ok").exists():
            err = unzip_avi(zip_path, avi)
            info["unzip_error"] = err
            if err is None:
                (raw_dir / name / ".ok").touch()
        info["avi"] = str(avi)
    n_video = probe_frames(avi)
    n_expected = (n_video + SAR_FPS - 1) // SAR_FPS
    frame_dir = out_dir / name
    done = frame_dir / ".frames_done"
    if done.exists() and not args.overwrite:
        prev = json.loads(done.read_text())
        n_jpg, n_err = prev["n_jpg"], prev["decode_errors"]
    else:
        n_jpg, n_err = extract_frames(avi, frame_dir)
        done.write_text(json.dumps({"n_jpg": n_jpg, "decode_errors": n_err}))
    info.update(n_video_frames=n_video, n_jpg=n_jpg, decode_errors=n_err,
                last_label_frame=int(frames.max()))
    reasons = []
    if info.get("unzip_error"):
        reasons.append(f"zip: {info['unzip_error']}")
    if n_err:
        reasons.append(f"{n_err} ffmpeg decode errors")
    if n_jpg != n_expected:
        reasons.append(f"{n_jpg} jpg extracted, {n_expected} expected")
    if frames.max() >= n_video:
        reasons.append("labels beyond the last video frame")
    info["exclude_reason"] = "; ".join(reasons) or None
    secs = np.arange(n_jpg, dtype=np.int32)
    g = gesture_per_second(frames, labels, n_jpg)
    info["n_seconds"] = int(n_jpg)
    info["n_unlabelled_seconds"] = int((g < 0).sum())
    save_video(out_dir, name, {
        "second": secs,
        "frame_id": secs.astype(np.int64) * SAR_FPS,
        "frame_path": np.array([str(frame_dir / f"{s:05d}.jpg") for s in secs]),
        "gestures": g,
    })
    return name, info, np.bincount(labels, minlength=len(SAR_GESTURES)), np.bincount(g[g >= 0], minlength=len(SAR_GESTURES))


def prepare_sar(out_root, args):
    out_dir = out_root / "sar_rarp50"
    raw_dir = out_dir / "_raw"
    items = [
        (split, f)
        for split in ("train", "test")
        for f in sorted((SAR_DIR / split).iterdir())
        if f.is_dir() and f.name.startswith("video_")
    ]
    if args.sar_videos:
        items = [it for it in items if it[1].name.removesuffix(".tmp") in args.sar_videos]
    meta = {"dataset": "sar_rarp50", "source": str(SAR_DIR), "vocab": {"gestures": SAR_GESTURES},
            "fps_video": SAR_FPS, "splits": {}, "videos": {}, "excluded": {}, "stats": {}}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda it: sar_video(it, out_dir, raw_dir, args), items))
    rows = collections.Counter()
    tot10 = collections.defaultdict(lambda: np.zeros(len(SAR_GESTURES), dtype=np.int64))
    tot1 = collections.defaultdict(lambda: np.zeros(len(SAR_GESTURES), dtype=np.int64))
    secs = collections.Counter()
    for name, info, c10, c1 in results:
        split = info["split"]
        meta["videos"][name] = info
        rows[split] += info["rows_10hz"]
        if info["exclude_reason"]:
            meta["excluded"][name] = info["exclude_reason"]
            continue
        meta["splits"].setdefault(split, []).append(name)
        secs[split] += info["n_seconds"]
        tot10[split] += c10
        tot1[split] += c1
    for split in ("train", "test"):
        meta["stats"][split] = {
            "videos": len(meta["splits"].get(split, [])),
            "rows_10hz_all_videos": rows[split],
            "seconds": secs[split],
            "per_class_rows_10hz": class_counts(SAR_GESTURES, tot10[split]),
            "per_class_seconds": class_counts(SAR_GESTURES, tot1[split]),
        }
        print(f"SAR {split}: {meta['stats'][split]['videos']} videos kept, {secs[split]} s, "
              f"{rows[split]} rows at 10 Hz (all folders)")
        print(f"    seconds per gesture: {meta['stats'][split]['per_class_seconds']}")
    print(f"SAR excluded: {meta['excluded']}")
    print(f"SAR frames: {time.time() - t0:.0f} s")
    ok = True
    if not args.sar_videos:
        ok = check("sar rows at 10 Hz", sum(rows.values()), EXPECTED["sar"]["rows_10hz"])
    meta["checks_ok"] = bool(ok)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return ok


# --------------------------------------------------------------------------- ESAD

ESAD_RE = re.compile(r"^(RARP\d+)_frame_(\d+)\.txt$")


def read_yolo(path, n_classes):
    ids = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            c = int(line.split()[0])
            assert 0 <= c < n_classes, (path, c)
            ids.append(c)
    return ids


def prepare_esad(out_root, args):
    out_dir = out_root / "esad"
    names = [l.strip() for l in (ESAD_DIR / "train/train/obj.names").read_text().splitlines() if l.strip()]
    assert len(names) == 21, len(names)
    meta = {"dataset": "esad", "source": str(ESAD_DIR), "vocab": {"actions": names}, "splits": {},
            "videos": {}, "excluded": {}, "stats": {}}
    ok = True
    for split, procs in ESAD_SPLITS.items():
        st = {"frames": 0, "frames_with_boxes": 0, "instances": 0, "missing_jpg": 0}
        inst = np.zeros(len(names), dtype=np.int64)
        frames_pos = np.zeros(len(names), dtype=np.int64)
        meta["splits"][split] = []
        for video, lab_rel, img_rel in procs:
            lab_dir, img_dir = ESAD_DIR / lab_rel, ESAD_DIR / img_rel
            entries = {}
            with os.scandir(lab_dir) as it:
                for e in it:
                    m = ESAD_RE.match(e.name)
                    if m:
                        entries[int(m.group(2))] = e.path
            with os.scandir(img_dir) as it:
                jpgs = {e.name for e in it if e.name.endswith(".jpg")}
            prefix = ESAD_RE.match(os.path.basename(next(iter(entries.values())))).group(1)
            secs = np.array(sorted(entries), dtype=np.int32)
            has_jpg = np.array([f"{prefix}_frame_{s}.jpg" in jpgs for s in secs])
            st["missing_jpg"] += int((~has_jpg).sum())
            secs = secs[has_jpg]
            with ThreadPoolExecutor(32) as ex:
                ids = list(ex.map(lambda s: read_yolo(entries[s], len(names)), secs))
            n_inst = np.array([len(i) for i in ids], dtype=np.int16)
            mh = multi_hot(ids, len(names), 0)
            save_video(out_dir, video, {
                "second": secs,
                "frame_id": secs.astype(np.int64),
                "frame_path": np.array([str(img_dir / f"{prefix}_frame_{s}.jpg") for s in secs]),
                "actions": mh,
                "n_instances": n_inst,
            })
            gaps = np.diff(secs)
            meta["videos"][video] = {"split": split, "challenge_name": prefix, "n_seconds": int(len(secs)),
                                     "first_second": int(secs[0]), "last_second": int(secs[-1]),
                                     "n_gaps": int((gaps > 1).sum()), "max_gap": int(gaps.max())}
            meta["splits"][split].append(video)
            st["frames"] += len(secs)
            st["frames_with_boxes"] += int((n_inst > 0).sum())
            st["instances"] += int(n_inst.sum())
            for i in ids:
                np.add.at(inst, i, 1)
            frames_pos += mh.sum(0, dtype=np.int64)
        st["per_class_instances"] = class_counts(names, inst)
        st["per_class_frames"] = class_counts(names, frames_pos)
        meta["stats"][split] = st
        print(f"ESAD {split}: {meta['splits'][split]}, {st['frames']} frames ({st['frames_with_boxes']} with boxes, "
              f"{st['missing_jpg']} txt without jpg), {st['instances']} instances")
        print(f"    instances per class: {st['per_class_instances']}")
        ok &= check(f"esad {split} instances", st["instances"], EXPECTED["esad"]["instances"][split])
    meta["checks_ok"] = bool(ok)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["grasp", "sar", "esad"], choices=["grasp", "sar", "esad"])
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--workers", type=int, default=6, help="concurrent ffmpeg processes (SAR)")
    parser.add_argument("--sar-videos", nargs="*", help="only these SAR videos (debug)")
    parser.add_argument("--overwrite", action="store_true", help="re-extract SAR frames")
    args = parser.parse_args()
    out_root = Path(args.out)
    assert not str(out_root.resolve()).startswith(str((WS / "nas_private").resolve())), "never write on the NAS"
    fns = {"grasp": prepare_grasp, "sar": prepare_sar, "esad": prepare_esad}
    results = {}
    for ds in args.datasets:
        t0 = time.time()
        results[ds] = fns[ds](out_root, args)
        print(f"== {ds}: {'checks OK' if results[ds] else 'CHECK MISMATCH'} ({time.time() - t0:.0f} s)\n", flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
