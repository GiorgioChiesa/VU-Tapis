#!/usr/bin/env python3
"""
Zero-shot event spotting on untrimmed Orsi videos with an open video-LLM
(Qwen3-VL family through native transformers classes).

Each video is cut into windows of `--window` seconds (stride `--stride`) at 1 fps.
For every window the model gets the frames, the English event catalogue
(`orsi_events_en.json`, a translation of analisi/2026-11-03_eventi_descrizione.md)
and the summary of the previous window, and must answer with JSON:
    {"summary": "...", "events": [{"event": <class name>, "time_s": <s in window>, "confidence": 0-1}]}

Outputs, in the format of `tools/infer_untrimmed.py` so that `tools/eval_untrimmed.py`
and `tools/event_confusion.py` work unchanged:
    <out-dir>/meta.json, <out-dir>/{patient}.npz  (second, frame_id, probs_steps [T, 33])
plus the textual log <out-dir>/log/{patient}.jsonl (one line per window: raw answer,
parsed events in absolute seconds, summary). Everything stays on this machine.

Runs in the `vlm` mamba env (transformers >= 4.57):
    /home/gchie/miniforge3/envs/vlm/bin/python tools/vlm/infer_vlm_events.py \
        --model /home/gchie/workspace/nas_private/models/Qwen__Qwen3-VL-8B-Instruct \
        --patients RARP49 --gpu 1 --out-dir outputs/vlm/qwen3vl8b/val
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REF_RUN = REPO / "outputs/orsi/LONG/control/totale"
VAL = ["RARP49", "RARP50", "RARP59", "RARP61", "RARP62", "RARP64"]


def class_names(run=REF_RUN):
    dist = pd.read_csv(run / "distributions/steps_distribution.csv", skipinitialspace=True)
    return [str(n).strip() for n in dist["name"]]


def build_prompt(catalogue, window, prev_summary):
    lines = [
        "You are an expert urologic surgeon annotating a robot-assisted radical prostatectomy video.",
        f"Scene: {catalogue['scene']}",
        f"The video clip has {window} frames sampled at 1 frame per second; frame k is at time k seconds "
        f"(0 to {window - 1}).",
        "Find every occurrence of the following surgical EVENTS in this clip. Each event is a single instant, "
        "placed at the moment given by its timestamp rule.",
        "",
    ]
    for e in catalogue["events"]:
        lines.append(f"- {e['name']} (phase: {e['phase']}): {e['what']} Timestamp: {e['timestamp']}. Cues: {e['cues']}")
    lines += [
        "",
        f"Context from the previous clip: {prev_summary or 'none (start of the video)'}",
        "",
        "Most clips contain NO event: then return an empty list. Only report an event if you clearly see it.",
        'Answer with JSON only, no other text: {"summary": "<one sentence: what the surgeon is doing and which '
        'phase this is>", "events": [{"event": "<exact event name from the list>", "time_s": <integer second '
        'in the clip>, "confidence": <0 to 1>}]}',
    ]
    return "\n".join(lines)


def parse_answer(text, names, window):
    """Extract (summary, [(name, t, conf)]) from the model answer; tolerant to code fences and junk."""
    start = text.find("{")
    if start < 0:
        return "", [], "no_json"
    try:  # first complete JSON object; ignores trailing junk such as an extra "}"
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return "", [], "bad_json"
    if not isinstance(obj, dict):
        return "", [], "bad_json"
    key = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
    lookup = {key(n): n for n in names}
    events = []
    for e in obj.get("events", []) or []:
        try:
            name = lookup.get(key(e.get("event", "")))
            t = float(e.get("time_s"))
            conf = float(e.get("confidence", 1.0))
        except (TypeError, ValueError, AttributeError):
            continue
        if name is None or name == "Idle" or not (0 <= t <= window - 1):
            continue
        events.append((name, t, min(max(conf, 0.0), 1.0)))
    return str(obj.get("summary", ""))[:500], events, "ok"


def frame_paths(video_dir, frame_ids):
    """
    jpg of each label frame_id, written by `tools/vlm/extract_frames.py`. The last label
    rows of some videos point past the end of the video: they reuse the previous frame
    (at most 5 rows back).
    """
    paths, missing = [], 0
    for i, f in enumerate(frame_ids):
        p = video_dir / f"{f:06d}.jpg"
        if not p.exists():
            missing += 1
            assert i > 0 and missing <= 5, f"{video_dir.name}: frame {f} missing, run tools/vlm/extract_frames.py"
            p = paths[-1]
        paths.append(p)
    return paths


def load_window(paths, max_side, workers=16):
    def one(p):
        im = Image.open(p).convert("RGB")
        s = max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        return np.asarray(im)

    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(one, paths)))  # [T, H, W, 3] uint8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--patients", nargs="*", default=VAL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-dir", default="/home/gchie/workspace/data/orsi_tensors")
    ap.add_argument("--image-dir", default="/home/gchie/workspace/data/orsi_frames448")
    ap.add_argument("--catalogue", default=str(HERE / "orsi_events_en.json"))
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--stride", type=int, default=45)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-seconds", type=int, default=0, help="only the first N seconds (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "log").mkdir(parents=True, exist_ok=True)
    names = class_names()
    idx = {n: i for i, n in enumerate(names)}
    catalogue = json.loads(Path(args.catalogue).read_text())
    assert {e["name"] for e in catalogue["events"]} == set(names) - {"Idle"}, "catalogue/class mismatch"

    meta = {
        "run": str(REF_RUN), "ckpt": "checkpoint_best_steps.pyth", "split": "val", "patients": args.patients,
        "tasks": ["steps"], "num_classes": [len(names)], "predictor": "vlm_zero_shot", "model": args.model,
        "window": args.window, "stride": args.stride, "max_side": args.max_side, "max_seconds": args.max_seconds,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    todo = {}
    for patient in args.patients:
        if (out_dir / f"{patient}.npz").exists() and not args.overwrite:
            print(f"{patient}: exists, skipping", flush=True)
            continue
        labels = pd.read_csv(Path(args.label_dir) / patient / "Label" / f"{patient}_all_labels.csv")
        if args.max_seconds:
            labels = labels.iloc[: args.max_seconds]
        todo[patient] = (labels, frame_paths(Path(args.image_dir) / patient, labels["frame_id"].to_numpy()))

    device = f"cuda:{args.gpu}"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    ).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    for patient, (labels, paths) in todo.items():
        target = out_dir / f"{patient}.npz"
        frame_ids = labels["frame_id"].to_numpy()
        n = len(labels)
        scores = np.zeros((n, len(names)), dtype=np.float32)
        starts = list(range(0, max(n - args.window, 0) + 1, args.stride))
        if starts[-1] + args.window < n:
            starts.append(n - args.window)
        prev, t0, stats = "", time.time(), {"ok": 0, "no_json": 0, "bad_json": 0, "events": 0}
        with open(out_dir / "log" / f"{patient}.jsonl", "w") as log:
            for k, s in enumerate(starts):
                w = min(args.window, n - s)
                frames = load_window(paths[s : s + w], args.max_side)
                msgs = [{"role": "user", "content": [
                    {"type": "video"}, {"type": "text", "text": build_prompt(catalogue, w, prev)}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(
                    text=[text], videos=[frames], return_tensors="pt",
                    videos_kwargs={"fps": 1.0, "do_sample_frames": False,
                                   "video_metadata": [{"fps": 1.0, "total_num_frames": w, "duration": float(w), "frames_indices": list(range(w))}]},
                ).to(device)
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                answer = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                summary, events, status = parse_answer(answer, names, w)
                stats[status] += 1
                stats["events"] += len(events)
                prev = summary or prev
                abs_events = []
                for name, t, conf in events:
                    sec = int(round(s + t))
                    scores[sec, idx[name]] = max(scores[sec, idx[name]], conf)
                    abs_events.append({"event": name, "second": sec, "confidence": conf})
                log.write(json.dumps({"start": s, "window": w, "status": status, "summary": summary,
                                      "events": abs_events, "raw": answer}) + "\n")
                log.flush()
                if k % 20 == 0:
                    el = time.time() - t0
                    print(f"{patient}: window {k + 1}/{len(starts)} ({el / (k + 1):.1f}s/window) {stats}", flush=True)

        # per-second probabilities: event confidences, Idle takes the rest (rows sum to 1)
        tot = scores.sum(axis=1, keepdims=True)
        probs = np.where(tot > 1, scores / np.maximum(tot, 1e-9), scores)
        probs[:, idx["Idle"]] = 1.0 - probs.sum(axis=1) + probs[:, idx["Idle"]]
        np.savez_compressed(target, second=labels["second"].to_numpy(), frame_id=frame_ids,
                            probs_steps=probs.astype(np.float32))
        print(f"{patient}: {n} s, {len(starts)} windows, {time.time() - t0:.0f}s, {stats}", flush=True)


if __name__ == "__main__":
    main()
