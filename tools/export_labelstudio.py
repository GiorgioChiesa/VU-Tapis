#!/usr/bin/env python3
"""
Label Studio view of the event-level errors exported by ``export_error_viewer.py``.

Every item of the error viewer (one matched / swapped / missed / false-alarm event)
becomes a Label Studio task:

  video      the 16 input frames of the encoder as an H.264 mp4 (``clip.mp4`` next to
             the jpgs of the viewer, ``--fps`` frames per second of playback)
  timeline   one region per frame range of the clip, prefixed by its source:
               "GT ▸ <event>"     ground-truth events inside the clip (1 frame)
               "MOD ▸ <event>"    decoded detections above the fold threshold (1 frame)
               "FASE GT ▸ <ph>"   ground-truth phase, run-length over the clip
               "FASE MOD ▸ <ph>"  argmax of the phase head, run-length over the clip
  data       filterable fields (tipo, evento_reale, evento_predetto, fase_reale,
             fase_predetta, paziente, fold, score, offset_s, eventi_*_nel_clip, ...)

The regions are imported as one annotation per task, so ground truth and model are
on the same timeline; ``verifica`` and ``note`` are left empty for the reviewer.

The frames come from the jpgs of ``--viewer`` (run ``export_error_viewer.py`` first),
never from the tensors, so the clips are exactly the ones of the html viewer.
Everything is written under ``--viewer`` (outside the repository: patient frames).

Example (Label Studio serving ``--files-root`` as local files, see docs in README):
    python tools/export_labelstudio.py --viewer /home/gchie/workspace/viewer/MSTCN_dilate3_ftenc_capchain \\
        --push http://127.0.0.1:8766 --token-file /home/gchie/workspace/viewer/labelstudio/token [--replace]
"""

import argparse
import html
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from export_error_viewer import MISSED, NO_EVENT, collect
from tapis.datasets import utils as data_utils

FFMPEG = str(Path(sys.executable).parent / "ffmpeg")
CONTROL = "timeline"
REVIEW = ["errore del modello", "errore di annotazione", "evento ambiguo", "predizione corretta", "non valutabile"]
COLORS = {"GT": "#2e9e44", "MOD": "#e8710a", "FASE GT": "#1a73e8", "FASE MOD": "#9334e6"}
TABS = [("Tutti", None), ("Errori", ("tipo", "not_equal", "corretto")), ("Scambi", ("tipo", "equal", "scambio")),
        ("Mancati", ("tipo", "equal", "mancato")), ("Falsi allarmi", ("tipo", "equal", "falso_allarme")),
        ("Duplicati", ("tipo", "equal", "duplicato")), ("Corretti", ("tipo", "equal", "corretto"))]
HIDDEN_COLUMNS = [f"tasks:{c}" for c in (
    "data.video", "data.info", "data.titolo", "data.fps", "annotations_results", "annotations_ids",
    "predictions_score", "predictions_model_versions", "predictions_results", "file_upload", "storage_filename",
    "created_at", "updated_at", "updated_by", "avg_lead_time", "draft_exists", "inner_id")]


def tag(src, name):
    return f"{src} ▸ {name}"


def label_config(events, phases):
    def labels(src, names):
        return "\n".join(f'      <Label value="{html.escape(tag(src, n))}" background="{COLORS[src]}"/>' for n in names)

    return f"""<View>
  <Header value="$titolo"/>
  <Video name="video" value="$video" frameRate="$fps"/>
  <TimelineLabels name="{CONTROL}" toName="video">
{labels("GT", events)}
{labels("MOD", events)}
{labels("FASE GT", phases)}
{labels("FASE MOD", phases)}
  </TimelineLabels>
  <Text name="info" value="$info"/>
  <Header value="Verifica" size="5"/>
  <Choices name="verifica" toName="video" choice="single-radio" showInline="true">
{chr(10).join(f'    <Choice value="{v}"/>' for v in REVIEW)}
  </Choices>
  <TextArea name="note" toName="video" placeholder="note" rows="2" editable="true"/>
</View>
"""


def runs(values):
    """[(value, first, last)] of consecutive equal values (0-based indices)."""
    out, start = [], 0
    for k in range(1, len(values) + 1):
        if k == len(values) or values[k] != values[start]:
            out.append((values[start], start, k - 1))
            start = k
    return out


def region(rid, label, first, last, score=None):
    r = {"id": rid, "from_name": CONTROL, "to_name": "video", "type": "timelinelabels", "origin": "manual",
         "value": {"ranges": [{"start": first + 1, "end": last + 1}], "timelinelabels": [label]}}
    if score is not None:
        r["meta"] = {"text": [f"score {score:.3f}"]}
    return r


def encode(d, fps):
    mp4 = d / "clip.mp4"
    if not mp4.exists():
        subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(fps), "-i", str(d / "%02d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                        str(mp4.with_suffix(".tmp.mp4"))], check=True)
        mp4.with_suffix(".tmp.mp4").rename(mp4)
    return mp4


def build_task(it, v, names, ph_names, files_root, url_prefix, fps):
    labels, clip = v["labels"], v["clip"]
    seq = data_utils.get_sequence(it["row"], clip["num_frames"] * clip["sampling_rate"], clip["sampling_rate"],
                                  len(labels), mode=clip["seq_mode"])
    secs = labels["second"].to_numpy()[seq]
    lo, hi = secs.min(), secs.max()

    def frame_of(t):
        return int(np.abs(secs - t).argmin())

    result = []
    gt_in = [(c, t) for c, t in v["gts"] if lo <= t <= hi]
    det_in = [(c, t, s) for c, t, s in v["dets"] if lo <= t <= hi]
    for k, (c, t) in enumerate(gt_in):
        f = frame_of(t)
        result.append(region(f"gt{k}", tag("GT", names[c]), f, f))
    for k, (c, t, s) in enumerate(det_in):
        f = frame_of(t)
        result.append(region(f"mod{k}", tag("MOD", names[c]), f, f, s))
    for src, ph in (("FASE GT", v["ph_gt"]), ("FASE MOD", v["ph_hat"])):
        vals = [int(ph[min(i, len(ph) - 1)]) for i in seq]
        for k, (p, a, b) in enumerate(runs(vals)):
            result.append(region(f"{src.replace(' ', '').lower()}{k}", tag(src, ph_names[p]), a, b))

    center = int(np.flatnonzero(np.asarray(seq) == it["row"])[0]) if it["row"] in seq else len(seq) // 2
    true = names[it["true"]] if it["true"] is not None else NO_EVENT
    pred = names[it["pred"]] if it["pred"] is not None else MISSED
    offset = None if it["t_gt"] is None or it["t_pred"] is None else float(it["t_pred"] - it["t_gt"])
    rel = it["dir"].relative_to(files_root)
    data = {
        "video": url_prefix + quote(str(rel / "clip.mp4")),
        "fps": fps,
        "titolo": f"{it['kind']}: {true} → {pred}  ({it['patient']}, fold {it['fold']}, "
                  f"frame centrale {center + 1}/{len(seq)} = {secs[center]} s)",
        "info": it["info"] + f"secondi del clip: {secs[0]}–{secs[-1]} ({fps:g} frame/s in riproduzione)\n",
        "tipo": it["kind"],
        "coppia": it["dir"].parent.name,
        "evento_reale": true,
        "evento_predetto": pred,
        "fase_reale": ph_names[v["ph_gt"][it["row"]]],
        "fase_predetta": ph_names[v["ph_hat"][it["row"]]],
        "paziente": it["patient"],
        "fold": int(it["fold"]),
        "secondo_centrale": int(secs[center]),
        "t_reale": it["t_gt"],
        "t_predetto": it["t_pred"],
        "offset_s": offset,
        "score": None if it["score"] is None else round(float(it["score"]), 3),
        "eventi_reali_nel_clip": ", ".join(sorted({names[c] for c, _ in gt_in})),
        "eventi_predetti_nel_clip": ", ".join(sorted({names[c] for c, _, _ in det_in})),
        "fasi_reali_nel_clip": ", ".join(sorted({ph_names[int(v["ph_gt"][min(i, len(v["ph_gt"]) - 1)])] for i in seq})),
        "fasi_predette_nel_clip": ", ".join(
            sorted({ph_names[int(v["ph_hat"][min(i, len(v["ph_hat"]) - 1)])] for i in seq})),
    }
    return {"data": data, "annotations": [{"result": result, "ground_truth": False}]}


def push(url, token, title, config, tasks, storage_path, replace=False):
    import requests

    s = requests.Session()
    s.headers["Authorization"] = f"Token {token}"
    for p in s.get(f"{url}/api/projects", params={"page_size": 1000}).json()["results"]:
        if p["title"] == title:
            assert replace, f"project {p['id']} '{title}' exists (it may hold reviews): pass --replace to delete it"
            s.delete(f"{url}/api/projects/{p['id']}").raise_for_status()
            print(f"deleted old project {p['id']}")
    r = s.post(f"{url}/api/projects", json={"title": title, "label_config": config,
                                            "description": "Errori per evento (±5 s), CV 5 fold, 50 video"})
    r.raise_for_status()
    pid = r.json()["id"]
    s.post(f"{url}/api/storages/localfiles", json={"project": pid, "path": storage_path, "use_blob_urls": False,
                                                    "title": "viewer"}).raise_for_status()
    for k in range(0, len(tasks), 500):
        s.post(f"{url}/api/projects/{pid}/import", json=tasks[k: k + 500]).raise_for_status()
    n = s.get(f"{url}/api/projects/{pid}").json()["task_number"]
    print(f"project {pid}: {n} tasks")
    # data manager tabs, one per kind of match; filters on any data.* column stay available
    for v in s.get(f"{url}/api/dm/views", params={"project": pid}).json():
        s.delete(f"{url}/api/dm/views/{v['id']}")
    hidden = {"explore": HIDDEN_COLUMNS, "labeling": HIDDEN_COLUMNS}
    for name, f in TABS:
        items = [] if f is None else [{"filter": f"filter:tasks:data.{f[0]}", "operator": f[1], "type": "String",
                                       "value": f[2]}]
        r = s.post(f"{url}/api/dm/views", json={"project": pid, "data": {
            "title": name, "filters": {"conjunction": "and", "items": items}, "hiddenColumns": hidden,
            "ordering": ["tasks:data.paziente"]}})
        r.raise_for_status()
        n = s.get(f"{url}/api/tasks", params={"project": pid, "view": r.json()["id"], "page_size": 1}).json()["total"]
        print(f"  tab {name}: {n}")
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temporal-root", default="outputs/orsi/TEMPORAL/cv_ft_dilate3")
    parser.add_argument("--frame-dir", default="/home/gchie/workspace/data/orsi_tensors")
    parser.add_argument("--viewer", required=True, help="--out of export_error_viewer.py")
    parser.add_argument("--files-root", default="/home/gchie/workspace/viewer",
                        help="host dir mounted as LOCAL_FILES_DOCUMENT_ROOT")
    parser.add_argument("--container-root", default="/label-studio/files", help="the same dir inside the container")
    parser.add_argument("--variant", default="caps_chain")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--phase-smooth", type=int, default=1)
    parser.add_argument("--gate-floor", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=2.0, help="playback frame rate of the mp4")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--push", help="Label Studio url: (re)create the project and import the tasks")
    parser.add_argument("--token-file")
    parser.add_argument("--replace", action="store_true", help="delete a project with the same title (and its reviews)")
    args = parser.parse_args()

    viewer, files_root = Path(args.viewer).resolve(), Path(args.files_root).resolve()
    col = collect(args.temporal_root, args.frame_dir, args.variant, args.alpha, args.phase_smooth, args.gate_floor,
                  args.delta, viewer)
    missing = [it["dir"] for it in col.items if not (it["dir"] / "15.jpg").exists()]
    assert not missing, f"{len(missing)} items without jpgs (e.g. {missing[0]}): run export_error_viewer.py first"

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda it: encode(it["dir"], args.fps), col.items))
    events = [n for n in col.names if n.lower() != "idle"]
    config = label_config(events, col.ph_names)
    tasks = [build_task(it, col.videos[it["patient"]], col.names, col.ph_names, files_root,
                        "/data/local-files/?d=", args.fps) for it in col.items]
    out = viewer / "labelstudio"
    out.mkdir(exist_ok=True)
    (out / "label_config.xml").write_text(config)
    (out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, default=float))
    print(f"{len(tasks)} tasks -> {out}", flush=True)

    if args.push:
        token = Path(args.token_file).read_text().strip()
        storage = str(Path(args.container_root) / viewer.relative_to(files_root) / "results")
        push(args.push.rstrip("/"), token, f"Errori {viewer.name}", config, tasks, storage, args.replace)


if __name__ == "__main__":
    main()
