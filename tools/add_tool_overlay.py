#!/usr/bin/env python3
"""
Add the surgical-tool boxes of the terzetto detector (SurgVU26 cat1) to an error viewer.

``export_error_viewer.py`` writes, for every item, the 16 clip frames the encoder saw
(224x224 crops of ``orsi_tensors/<p>/Video_1fps``). The tool labels
(``nas_private/data/Orsi/tools_labels/<p>_tools.json``, 1 FPS) are in pixels of the
original video. For every second of every clip this script re-decodes the original frame
exactly as ``data/check_webapp/main.py:process_video`` did (seek by ms, black-border
crop, short side to 224, centre crop), maps the boxes into the 224 crop, and checks the
geometry by rebuilding the tensor and comparing it with the stored ``.pt``.

It writes, without touching the jpgs:
  <viewer>/tools_overlay.js                      toggle button (key B) + SVG overlay
  <viewer>/tool_overlay_summary.json             counts and geometry check
  results/<pair>/<item>/tools.json                boxes per clip frame (224 coordinates)
  results/<pair>/<item>/index.html                the 16 frames, 2x, with the overlay
and adds the overlay script to every results/<pair>/index.html and a note to index.html.
Re-run it after every ``export_error_viewer.py``.

Example (Vu-Tapis env, from src/VU-Tapis):
    python tools/add_tool_overlay.py --viewer /home/gchie/workspace/viewer/MSTCN_dilate3_ftenc_capchain \
        --temporal-root outputs/orsi/TEMPORAL/cv_ft_dilate3
"""

import argparse
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageChops

from tapis.datasets import utils as data_utils

SIDE = 224
MARK = "<!-- tool-overlay -->"
ITEM_RE = re.compile(r"(RARP\d+)_t(\d+)_fold(\d+)$")
MAE_TOL = 0.02  # rebuilt vs stored tensor, values in [0, 1]


def border_bbox(im):
    """The crop of ``data/removeBlackBorders.removeBlackBorders`` (same test, same bbox)."""
    threshold = 10
    if im.getpixel((0, 0)) <= (threshold, threshold, threshold):
        diff = ImageChops.difference(im, Image.new(im.mode, im.size, (0, 0, 0)))
        diff = ImageChops.add(diff, diff, 2.0, -100)
        return diff.getbbox()
    return None


def geometry(frame):
    """Crop/scale of process_video for one decoded frame, and the rebuilt 224 tensor."""
    im = Image.fromarray(frame)  # BGR array, as to_pil_image(frame) there
    bb = border_bbox(im) or (0, 0, im.width, im.height)
    im = im.crop(bb)
    w, h = im.size
    if (w <= h and w == SIDE) or (h <= w and h == SIDE):  # preprocess_frames.scale
        nw, nh = w, h
    elif w < h:
        nw, nh = SIDE, int(math.floor(h / w * SIDE))
    else:
        nw, nh = int(math.floor(w / h * SIDE)), SIDE
    im = im.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - SIDE) / 2, (nh - SIDE) / 2
    box = tuple(int(round(v)) for v in (left, top, left + SIDE, top + SIDE))  # PIL rounds crop boxes
    im = im.crop(box)
    tensor = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    return dict(x0=bb[0], y0=bb[1], sx=nw / w, sy=nh / h, left=box[0], top=box[1], border=list(bb)), tensor


def to_crop(xyxy, g):
    x1, y1, x2, y2 = xyxy
    x1, x2 = ((v - g["x0"]) * g["sx"] - g["left"] for v in (x1, x2))
    y1, y2 = ((v - g["y0"]) * g["sy"] - g["top"] for v in (y1, y2))
    x1, x2 = max(0.0, x1), min(float(SIDE), x2)
    y1, y2 = max(0.0, y1), min(float(SIDE), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def find_video(root, p):
    for name in (f"{p}.mp4", f"{p}_h264.mp4"):
        path = root / p / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no video for {p} under {root}")


def video_worker(p, seconds, frame_names, videos_root, tensor_dir, tools_dir):
    """Boxes in crop coordinates for each needed second of one video, plus the geometry check."""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    doc = json.loads((Path(tools_dir) / f"{p}_tools.json").read_text())
    frames = doc["frames"]
    # shared host: cap ffmpeg's own decode threads (it uses every core otherwise)
    cap = cv2.VideoCapture(str(find_video(Path(videos_root), p)), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    out, maes, borders = {}, [], Counter()
    for s in seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ok, frame = cap.read()
        if not ok:
            out[s] = dict(boxes=[], status="undecodable")
            continue
        g, tensor = geometry(frame)
        stored = torch.load(Path(tensor_dir) / p / "Video_1fps" / frame_names[s], map_location="cpu",
                            weights_only=True).float()
        mae = float((tensor - stored).abs().mean()) if stored.shape == tensor.shape else 1.0
        maes.append(mae)
        borders[tuple(g["border"])] += 1
        rec = frames[s] if s < len(frames) else None
        if rec is None or rec["second"] != s:
            out[s] = dict(boxes=[], status="no_label", mae=mae)
            continue
        boxes = []
        for b in rec["boxes"]:
            xyxy = to_crop(b["bbox_xyxy"], g)
            if xyxy:
                boxes.append(dict(tool=b["tool"], conf=round(b["confidence"], 3), xyxy=xyxy,
                                  above=bool(b["above_threshold"])))
        out[s] = dict(boxes=boxes, status="ok", mae=mae)
    cap.release()
    return p, out, maes, len(borders)


def clip_seconds(item_dir, labels, meta):
    p, center, _ = ITEM_RE.match(item_dir.name).groups()
    secs = labels["second"].to_numpy()
    row = min(int(np.searchsorted(secs, int(center))), len(labels) - 1)  # as export_error_viewer.collect
    rate = meta["sampling_rate"]
    seq = data_utils.get_sequence(row, meta["num_frames"] * rate, rate, len(labels), mode=meta["seq_mode"])
    return [(int(secs[idx]), Path(labels["frame_path"].iloc[idx]).name, idx - row) for idx in seq]


def item_page(item_dir, pair, frames):
    info = (item_dir / "info.txt").read_text() if (item_dir / "info.txt").exists() else ""
    cells = []
    for k, f in enumerate(frames):
        tools = ", ".join(f"{b['tool']} {b['conf']:.2f}" for b in f["boxes"] if b["above"]) or "nessuno strumento"
        cells.append(f'<div class="fr" data-k="{k}"><div class="to-wrap"><img src="{k:02d}.jpg"></div>'
                     f'<div class="cap">t={f["second"]} s ({f["offset"]:+d})</div>'
                     f'<div class="cap to-txt">{html.escape(tools)}</div></div>')
    (item_dir / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{item_dir.name}</title>
<style>body{{font-family:sans-serif;margin:16px}} .fr{{display:inline-block;margin:4px;vertical-align:top;width:448px}}
.fr img{{width:448px;display:block}} .cap{{font-size:12px}} pre{{font-size:12px}}</style></head><body>
<p><a href="../index.html">{html.escape(pair)}</a> · <a href="../../../index.html">indice</a> · <a href="strip.jpg">strip</a></p>
<pre>{html.escape(info)}</pre>
<p>Riquadro giallo nello strip = secondo dell'inferenza. Le bbox sono le detection del terzetto SurgVU26, non annotazioni.</p>
{"".join(cells)}
{MARK}<script src="../../../tools_overlay.js"></script></body></html>""")


def patch(path, before, snippet):
    text = path.read_text()
    if MARK in text:
        return False
    path.write_text(text.replace(before, f"{MARK}{snippet}{before}", 1))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viewer", required=True)
    parser.add_argument("--temporal-root", default="outputs/orsi/TEMPORAL/cv_ft_dilate3")
    parser.add_argument("--tensor-dir", default="/home/gchie/workspace/data/orsi_tensors")
    parser.add_argument("--videos-root", default="/home/gchie/workspace/nas_private/data/Orsi/videos")
    parser.add_argument("--tools-dir", default="/home/gchie/workspace/nas_private/data/Orsi/tools_labels")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    viewer = Path(args.viewer)
    metas = {fd.name.removeprefix("fold"): json.loads((fd / "val" / "meta.json").read_text())
             for fd in sorted(Path(args.temporal_root).glob("fold*"))}
    items = sorted(d for d in (viewer / "results").glob("*/*") if d.is_dir() and ITEM_RE.match(d.name))
    labels_of, clips, needed = {}, {}, defaultdict(dict)
    for d in items:
        p, _, fold = ITEM_RE.match(d.name).groups()
        if p not in labels_of:
            labels_of[p] = pd.read_csv(Path(args.tensor_dir) / p / "Label" / f"{p}_all_labels.csv",
                                       usecols=["second", "frame_path"])
        clips[d] = clip_seconds(d, labels_of[p], metas[fold])
        for s, name, _ in clips[d]:
            needed[p][s] = name
    print(f"[overlay] {len(items)} items, {len(needed)} videos, "
          f"{sum(len(v) for v in needed.values())} distinct seconds", flush=True)

    boxes_of, maes, status = {}, [], Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(video_worker, p, sorted(secs), secs, args.videos_root, args.tensor_dir, args.tools_dir)
                for p, secs in needed.items()]
        for fut in futs:
            p, out, m, n_borders = fut.result()
            boxes_of[p] = out
            maes += m
            status.update(r["status"] for r in out.values())
            bad = sum(v > MAE_TOL for v in m)
            print(f"[overlay] {p}: {len(out)} s, mae max {max(m, default=0):.4f}, >{MAE_TOL}: {bad}, "
                  f"distinct border crops {n_borders}", flush=True)

    maes = np.array(maes)
    n_bad = int((maes > MAE_TOL).sum())
    summary = dict(items=len(items), seconds=int(len(maes)), status=dict(status),
                   mae_median=float(np.median(maes)), mae_p99=float(np.percentile(maes, 99)),
                   mae_max=float(maes.max()), over_tolerance=n_bad, tolerance=MAE_TOL)
    print(f"[overlay] geometry check: {json.dumps(summary)}", flush=True)
    assert n_bad <= 0.005 * len(maes), "rebuilt crops differ from the stored tensors: geometry is wrong"

    pairs = defaultdict(list)
    for d, clip in clips.items():
        p = ITEM_RE.match(d.name).group(1)
        frames = [dict(k=k, second=s, offset=off, boxes=boxes_of[p][s]["boxes"]) for k, (s, _, off) in enumerate(clip)]
        center_k = next((k for k, f in enumerate(frames) if f["offset"] == 0), len(frames) // 2)
        in_clip = Counter(t for f in frames for t in {b["tool"] for b in f["boxes"] if b["above"]})  # frames per tool
        (d / "tools.json").write_text(json.dumps(dict(threshold=0.3, center_k=center_k, in_clip=dict(in_clip),
                                                      frames=frames)))
        item_page(d, d.parent.name, frames)
        pairs[d.parent].append(d)
    patched = sum(patch(pair / "index.html", "</body>", '<script src="../../tools_overlay.js"></script>')
                  for pair in pairs if (pair / "index.html").exists())
    patch(viewer / "index.html", "<img src=\"confusion_matrix.png\">",
          "<p><b>Bbox strumenti</b>: detection del terzetto SurgVU26 (YOLOv8n + 2×RT-DETR-L, WBF+TTA), "
          "non annotazioni; soglia 0.3. Pulsante in alto a destra o tasto <b>B</b> per mostrarle/nasconderle, "
          "casella per vedere anche i box sotto soglia (tratteggiati). Cliccando uno strip si apre il clip a 2×.</p>")
    js = Path(__file__).with_name("tools_overlay.js")
    (viewer / "tools_overlay.js").write_text(js.read_text())
    summary["pair_pages_patched"] = patched
    (viewer / "tool_overlay_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[overlay] wrote {len(clips)} tools.json/index.html, patched {patched} pair pages", flush=True)


if __name__ == "__main__":
    main()
