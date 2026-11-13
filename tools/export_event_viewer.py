#!/usr/bin/env python3
"""
Event view of an error viewer: one folder per ground-truth event, served on its own port.

``export_error_viewer.py`` groups the clips by ``<true>--<pred>`` pair. This script
regroups the same clips by the **true** event (false alarms go to ``nessun_evento``),
without copying frames: every clip is a symlink to its item folder, whose 16 jpgs,
``info.txt`` and ``tools.json`` (``add_tool_overlay.py``) are reused as they are.

  <out>/index.html                       the events, with clips per outcome
  <out>/event_viewer.js                  bbox toggle (key B / L), outcome filter, video chapters
  <out>/<Event>/index.html               montage on top, then one 4x4 grid + info per clip
  <out>/<Event>/montaggio.mp4            all clips in a row, 2x, tool boxes and prediction burnt in
  <out>/<Event>/chapters.json            start of every clip in the montage
  <out>/<Event>/clips/<item> -> ../../<viewer>/results/<pair>/<item>

Serve it with ``tools/serve_viewer.py`` (HTTP Range support, needed to seek in the mp4).

Example (Vu-Tapis env, from src/VU-Tapis):
    python tools/export_event_viewer.py --viewer /home/gchie/workspace/viewer/MSTCN_dilate3_ftenc_capchain \
        --out /home/gchie/workspace/viewer/eventi_MSTCN_dilate3_ftenc_capchain
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FFMPEG = str(Path(sys.executable).parent / "ffmpeg")
FONT = str(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
ITEM_RE = re.compile(r"(RARP\d+)_t(\d+)_fold(\d+)$")
NO_EVENT = "nessun_evento"
SIDE, SCALE = 224, 2
OUTCOMES = {  # info.txt "tipo" -> label, colour
    "corretto": ("corretto", "#1baf7a"),
    "scambio": ("scambio", "#e34948"),
    "mancato": ("mancato", "#8a8a8a"),
    "duplicato": ("duplicato", "#9467bd"),
    "falso_allarme": ("falso allarme", "#eb6834"),
}
# Same colours as the terzetto service and tools_overlay.js
CLASSES = ["grasping_retractor", "cadiere_forceps", "bipolar_forceps", "force_bipolar", "clip_applier", "stapler",
           "permanent_cautery_hook_spatula", "monopolar_curved_scissor", "vessel_sealer",
           "tip_up_fenestrated_grasper", "bipolar_dissector", "needle_driver", "prograsp_forceps",
           "suction_irrigator"]
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
           "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
COLOR = dict(zip(CLASSES, PALETTE))


def parse_info(text):
    f = dict(re.findall(r"^([^:\n]+?)(?: @ \S+)?: (.*)$", text, flags=re.M))
    true = re.match(r"(.*) @ (\S+) s$", f["evento reale"])
    pred = re.match(r"(.*) @ (\S+) s\s+score (\S+)$", f["evento predetto"])
    num = lambda v: None if v == "-" else float(v)  # noqa: E731
    return dict(kind=f["tipo"], true=true.group(1), t_gt=num(true.group(2)), pred=pred.group(1),
                t_pred=num(pred.group(2)), score=num(pred.group(3)),
                offset=num(f["offset predetto-reale"].removesuffix(" s")),
                phase_gt=f.get("fase reale", "-"), phase_pred=f.get("fase predetta", "-"))


def fmt_s(v):
    return "-" if v is None else f"{v:g}"


def prediction_text(c):
    """The two lines burnt into the video and shown on the grid."""
    label = OUTCOMES[c["kind"]][0].upper()
    if c["kind"] == "mancato":
        line1 = f"{label} · nessuna predizione"
    else:
        line1 = f"{label} · predetto: {c['pred']} · score {fmt_s(c['score'])}"
        if c["offset"] is not None:
            line1 += f" · offset {c['offset']:+g} s"
    return line1


def collect(viewer):
    clips = []
    for d in sorted((viewer / "results").glob("*/*")):
        m = ITEM_RE.match(d.name)
        if not (d.is_dir() and m):
            continue
        info = parse_info((d / "info.txt").read_text())
        event = d.parent.name.split("--")[0]
        assert event == info["true"] or event == NO_EVENT == info["true"], (d, info["true"])
        clips.append(dict(info, event=event, name=d.name, src=d, patient=m.group(1), center=int(m.group(2)),
                          fold=m.group(3), pair=d.parent.name))
    return clips


def draw_frame(img, boxes, c, second, k, n, offset, font, small):
    """One 448x448 montage frame: tool boxes >= 0.3, centre frame border, prediction box top-left."""
    img = img.resize((SIDE * SCALE, SIDE * SCALE), Image.BILINEAR).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    line1 = prediction_text(c)
    line2 = f"{c['patient']} · t={second} s · frame {k + 1}/{n}" + (" · centro" if offset == 0 else "")
    w = max(d.textlength(line1, font=font), d.textlength(line2, font=small)) + 12
    pred_box = (0, 0, min(w, img.width), 36)
    d.rectangle(pred_box, fill=(0, 0, 0, 170))
    d.rectangle((0, 0, 5, 36), fill=OUTCOMES[c["kind"]][1])
    d.text((10, 3), line1, font=font, fill=(255, 255, 255, 255))
    d.text((10, 20), line2, font=small, fill=(220, 220, 220, 255))
    for b in sorted(boxes, key=lambda b: b["conf"]):
        if not b["above"]:
            continue
        x1, y1, x2, y2 = (v * SCALE for v in b["xyxy"])
        col = COLOR.get(b["tool"], "#ffff00")
        d.rectangle((x1, y1, x2, y2), outline=col, width=2)
        label = f"{b['tool']} {b['conf']:.2f}"
        tw = d.textlength(label, font=small) + 6
        ly = y1 - 15 if y1 >= 15 else y1 + 1
        # keep the prediction readable: labels that would cover it go under the box
        if x1 < pred_box[2] and ly < pred_box[3]:
            ly = min(y2 + 1, img.height - 15)
        d.rectangle((x1, ly, x1 + tw, ly + 14), fill=col)
        d.text((x1 + 3, ly + 1), label, font=small, fill=(255, 255, 255, 255))
    if offset == 0:
        d.rectangle((0, 0, img.width - 1, img.height - 1), outline=(255, 210, 0, 255), width=4)
    return Image.alpha_composite(img, over).convert("RGB")


def render_montage(event, clips, out_dir, fps, force):
    """Writes montaggio.mp4 + chapters.json for one event; returns (event, n_clips, seconds)."""
    mp4, chapters = out_dir / "montaggio.mp4", out_dir / "chapters.json"
    if mp4.exists() and chapters.exists() and not force:
        return event, len(clips), json.loads(chapters.read_text())["duration"]
    font, small = ImageFont.truetype(FONT, 13), ImageFont.truetype(FONT, 11)
    size = SIDE * SCALE
    tmp = mp4.with_suffix(".tmp.mp4")
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-threads", "2", str(tmp)], stdin=subprocess.PIPE)
    chap, n_frames = [], 0
    for c in clips:
        tools = json.loads((c["src"] / "tools.json").read_text())
        frames = tools["frames"]
        chap.append(dict(clip=c["link"], start=n_frames / fps, kind=c["kind"]))
        for k, f in enumerate(frames):
            img = Image.open(c["src"] / f"{k:02d}.jpg").convert("RGB")
            frame = draw_frame(img, f["boxes"], c, f["second"], k, len(frames), f["offset"], font, small)
            proc.stdin.write(frame.tobytes())
            n_frames += 1
    proc.stdin.close()
    assert proc.wait() == 0, f"ffmpeg failed for {event}"
    tmp.replace(mp4)
    chapters.write_text(json.dumps(dict(fps=fps, duration=n_frames / fps, chapters=chap)))
    return event, len(clips), n_frames / fps


def clip_card(i, c, start):
    color = OUTCOMES[c["kind"]][1]
    imgs = "".join(f'<img loading="lazy" src="clips/{c["link"]}/{k:02d}.jpg" data-k="{k}">' for k in range(16))
    pred = "—" if c["kind"] == "mancato" else (
        f'{html.escape(c["pred"])} · score {fmt_s(c["score"])}'
        + ("" if c["offset"] is None else f' · offset {c["offset"]:+g} s'))
    t_gt = "—" if c["t_gt"] is None else f'{c["t_gt"]:g} s'
    t_pred = "—" if c["t_pred"] is None else f'{c["t_pred"]:g} s'
    return f"""<section class="clip" data-kind="{c['kind']}" data-dir="clips/{c['link']}/" id="c{i}">
<div class="grid-wrap"><div class="grid">{imgs}</div>
<div class="pred" style="border-left-color:{color}"><b style="color:{color}">{OUTCOMES[c['kind']][0].upper()}</b> · {pred}</div></div>
<div class="meta"><div><b>{c['patient']}</b> · fold {c['fold']} · secondo centrale {c['center']} s ·
<a href="#" class="seek" data-start="{start:g}">▶ nel video</a> · <a href="clips/{c['link']}/">cartella</a></div>
<div>evento reale: <b>{html.escape(c['true'].replace('_', ' '))}</b> @ {t_gt} · predizione: <b>{pred}</b> @ {t_pred}</div>
<div>fase reale: {html.escape(c['phase_gt'])} · fase predetta: {html.escape(c['phase_pred'])}</div>
<div class="to-txt"></div></div></section>"""


def event_page(out_dir, event, clips, chapters):
    counts = Counter(c["kind"] for c in clips)
    filters = "".join(
        f'<label class="flt"><input type="checkbox" value="{k}" checked> '
        f'<span style="color:{OUTCOMES[k][1]}">■</span> {OUTCOMES[k][0]} ({counts[k]})</label>'
        for k in OUTCOMES if counts[k])
    cards = "\n".join(clip_card(i, c, ch["start"]) for i, (c, ch) in enumerate(zip(clips, chapters["chapters"])))
    title = event.replace("_", " ")
    (out_dir / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="../event_viewer.css"></head><body>
<p><a href="../index.html">← tutti gli eventi</a></p>
<h1>{html.escape(title)} <small>({len(clips)} clip)</small></h1>
<div class="player"><video id="montage" src="montaggio.mp4" controls preload="metadata"></video>
<div class="pctl">velocità <button data-rate="0.5">×0.5</button><button data-rate="1">×1</button>
<button data-rate="2">×2</button><button data-rate="4">×4</button>
<span id="now"></span></div>
<p class="note">Montaggio di tutti i clip dell'evento a {chapters['fps']:g} fps (ogni clip 16 frame = {16 / chapters['fps']:g} s),
con le bbox degli strumenti ≥ 0.3 e la predizione in alto a sinistra. Durata {chapters['duration'] / 60:.1f} min.</p></div>
<div class="bar"><span class="filters">{filters}</span></div>
{cards}
<script>window.CHAPTERS = {json.dumps(chapters['chapters'])};</script>
<script src="../event_viewer.js"></script></body></html>""")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viewer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="re-render existing montages")
    args = parser.parse_args()

    viewer, out = Path(args.viewer).resolve(), Path(args.out).resolve()
    clips = collect(viewer)
    by_event = defaultdict(list)
    for c in clips:
        by_event[c["event"]].append(c)
    n_items = sum(1 for d in (viewer / "results").glob("*/*") if d.is_dir() and ITEM_RE.match(d.name))
    assert len(clips) == n_items and len({c["src"] for c in clips}) == len(clips), "clips lost or repeated"
    print(f"[eventi] {len(clips)} clips, {len(by_event)} folders: "
          + ", ".join(f"{e} {len(v)}" for e, v in sorted(by_event.items(), key=lambda kv: -len(kv[1]))), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    for name in ("event_viewer.js", "event_viewer.css"):
        shutil.copyfile(here / name, out / name)
    for event, cl in by_event.items():
        cl.sort(key=lambda c: (c["patient"], c["center"]))
        names = Counter(c["name"] for c in cl)
        for c in cl:  # two outcomes can share a second (e.g. a hit and a swap): keep both
            c["link"] = c["name"] if names[c["name"]] == 1 else f'{c["name"]}_{c["kind"]}'
        assert len({c["link"] for c in cl}) == len(cl), event
        links = out / event / "clips"
        links.mkdir(parents=True, exist_ok=True)
        for c in cl:
            link = links / c["link"]
            target = os.path.relpath(c["src"], links)
            if link.is_symlink() and os.readlink(link) != target:
                link.unlink()
            if not link.is_symlink():
                link.symlink_to(target)

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(render_montage, e, cl, out / e, args.fps, args.force)
                for e, cl in sorted(by_event.items(), key=lambda kv: -len(kv[1]))]
        for fut in futs:
            event, n, dur = fut.result()
            assert abs(dur - n * 16 / args.fps) < 1e-6, (event, n, dur)
            print(f"[eventi] {event}: {n} clips, montage {dur / 60:.1f} min", flush=True)
            chapters = json.loads((out / event / "chapters.json").read_text())
            event_page(out / event, event, by_event[event], chapters)
            rows.append((event, Counter(c["kind"] for c in by_event[event]), dur))

    head = "".join(f'<th style="color:{col}">{lab}</th>' for lab, col in OUTCOMES.values())
    body = "\n".join(
        f'<tr><td><a href="{e}/index.html">{html.escape(e.replace("_", " "))}</a></td><td>{sum(cnt.values())}</td>'
        + "".join(f"<td>{cnt[k] or ''}</td>" for k in OUTCOMES) + f"<td>{dur / 60:.1f}</td></tr>"
        for e, cnt, dur in sorted(rows, key=lambda r: (r[0] == NO_EVENT, -sum(r[1].values()))))
    (out / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Eventi · {viewer.name}</title>
<link rel="stylesheet" href="event_viewer.css"></head><body>
<h1>Vista per evento · {html.escape(viewer.name)}</h1>
<p>Gli stessi {len(clips)} clip del viewer degli errori (porta 8765), raggruppati per <b>evento reale</b>; i falsi allarmi
sono in <i>nessun evento</i>. Ogni pagina ha in alto il montaggio mp4 di tutti i clip ({args.fps:g} fps), poi la griglia
4×4 dei 16 frame a 1 fps di ogni clip con la predizione e le informazioni. <b>Bbox strumenti</b>: detection del terzetto
SurgVU26 (YOLOv8n + 2×RT-DETR-L, WBF+TTA), non annotazioni; soglia 0.3; pulsante o tasto <b>B</b>, tasto <b>L</b> per i
box sotto soglia.</p>
<table class="ev"><tr><th>evento reale</th><th>clip</th>{head}<th>video (min)</th></tr>
{body}</table></body></html>""")
    print(f"[eventi] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
