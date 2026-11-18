#!/usr/bin/env python3
"""
EXP-19 stage C: second-by-second inference with the causal MS-TCN and the online decoder.

For every validation video of a fold:
  * the saved clip features are read one second at a time (the encoder clip of second t
    already contains 7 s of future frames, so the declared latency is 7 s + L);
  * the causal MS-TCN is advanced by one step (`StreamingTemporalModel`: per-layer buffers of
    the last 2d inputs, no access to later seconds). With --check-forward the streamed scores
    are compared with a full forward pass of the same model;
  * the online decoder (`tapis.evaluate.online_decode.StreamDecoder`) with the per-class
    parameters chosen in stage B emits events;
  * an event log ``{class, t_event, t_emit, score, alarm}`` is written per video (stays on Orsi).

Example:
    python tools/stream_infer.py --fold 0 --variant pc_chain --budget 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np

from tapis.evaluate.online_decode import StreamDecoder


def _np(t):
    return t.detach().cpu().numpy().astype(np.float64)


class StreamingTemporalModel:
    """
    Step-by-step evaluation of a causal `train_temporal.TemporalModel` (eval mode).

    Each dilated layer keeps its last ``2d`` inputs in a ring buffer; zeros before the first
    second reproduce the causal zero padding ``(2d, 0)``, so ``step`` returns the same scores
    as the full forward pass, one second at a time.
    """

    def __init__(self, model):
        self.stages = []
        for stage in model.stages:
            layers = []
            for layer in stage.layers:
                if layer.pad[1] != 0:
                    raise ValueError("StreamingTemporalModel needs a causal model (--causal)")
                d = layer.conv.dilation[0]
                layers.append({
                    "d": d,
                    "w": _np(layer.conv.weight),  # [C, C, 3]: taps at t-2d, t-d, t
                    "b": _np(layer.conv.bias),
                    "wo": _np(layer.out.weight)[:, :, 0],
                    "bo": _np(layer.out.bias),
                })
            self.stages.append({
                "wi": _np(stage.inp.weight)[:, :, 0], "bi": _np(stage.inp.bias),
                "wp": _np(stage.phase.weight)[:, :, 0], "bp": _np(stage.phase.bias),
                "we": _np(stage.event.weight)[:, :, 0], "be": _np(stage.event.bias),
                "layers": layers,
            })
        self.reset()

    def reset(self):
        self.t = 0
        for st in self.stages:
            for ly in st["layers"]:
                ly["buf"] = np.zeros((2 * ly["d"] + 1, len(ly["b"])))

    def step(self, x):
        """``x``: normalised feature of the current second [D] -> (phase logits, event logits)."""
        inp = np.asarray(x, dtype=np.float64)
        for k, st in enumerate(self.stages):
            if k:
                ph, ev = out
                e = np.exp(ph - ph.max())
                inp = np.concatenate([e / e.sum(), 1.0 / (1.0 + np.exp(-ev))])
            h = st["wi"] @ inp + st["bi"]
            for ly in st["layers"]:
                d, buf = ly["d"], ly["buf"]
                n = len(buf)
                buf[self.t % n] = h
                past1 = buf[(self.t - d) % n] if self.t >= d else np.zeros_like(h)
                past2 = buf[(self.t - 2 * d) % n] if self.t >= 2 * d else np.zeros_like(h)
                w = ly["w"]
                a = w[:, :, 0] @ past2 + w[:, :, 1] @ past1 + w[:, :, 2] @ h + ly["b"]
                h = h + ly["wo"] @ np.maximum(a, 0.0) + ly["bo"]
            out = (st["wp"] @ h + st["bp"], st["we"] @ h + st["be"])
        self.t += 1
        return out


def event_probs(ev_logits, bg, fg):
    """Same transform as `train_temporal.predict`: sigmoid, background = 1 - max foreground."""
    p = 1.0 / (1.0 + np.exp(-ev_logits))
    if bg:
        p[list(bg)] = 1.0 - p[fg].max()
    return p


def load_params(path):
    """Stage-B parameter file -> keyword arguments of StreamDecoder / decode_online."""
    raw = json.loads(Path(path).read_text())
    per_class = lambda d: {int(k): v for k, v in d.items()} if isinstance(d, dict) else d  # noqa: E731
    return {
        "theta": per_class(raw["theta"]),
        "window": per_class(raw["window"]),
        "delay": per_class(raw["delay"]),
        "caps": per_class(raw.get("caps") or {}),
        "chain": raw.get("chain", []),
        "alpha": raw.get("alpha", 0.1),
        "chain_rule": raw.get("chain_rule", "order"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--a1-root", default="outputs/orsi/EXP19/A1_causal_dilate3")
    parser.add_argument("--params-dir", default="outputs/orsi/EXP19/pp_params")
    parser.add_argument("--variant", default="pc_chain")
    parser.add_argument("--budget", type=int, default=5, help="latency budget L (s) of the stage-B parameters")
    parser.add_argument("--out", default="outputs/orsi/EXP19/C_logs")
    parser.add_argument("--check-forward", action="store_true", help="compare with a full forward pass")
    args = parser.parse_args()

    import torch

    from infer_temporal import load_temporal

    torch.backends.cudnn.allow_tf32 = False
    fold_dir = Path(args.a1_root) / f"fold{args.fold}"
    head = load_temporal(fold_dir, "cpu")
    stream = StreamingTemporalModel(head["model"])
    params = load_params(Path(args.params_dir) / f"{args.variant}_L{args.budget}_fold{args.fold}.json")
    meta = json.loads((fold_dir / "val" / "meta.json").read_text())
    out = Path(args.out) / f"{args.variant}_L{args.budget}" / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(head["args"]["feat_dir"])
    names = head["names"]
    report = {}
    for p in meta["patients"]:
        dst = out / f"{p}.json"
        if dst.exists():
            continue
        npz = np.load(feat_dir / f"{p}.npz")
        x = (npz["features"].astype(np.float32) - head["mu"]) / head["sd"]
        sec = npz["second"]
        dec = StreamDecoder(len(names), background_classes=head["bg"], **params)
        stream.reset()
        events, rows = [], []
        t0 = time.time()
        for t in range(len(x)):
            _, ev = stream.step(x[t])
            row = event_probs(ev, head["bg"], head["fg"])
            rows.append(row)
            events += dec.step(row)
        events += dec.flush()
        elapsed = time.time() - t0
        info = {"seconds": len(x), "wall_s": round(elapsed, 2)}
        if args.check_forward:
            from infer_temporal import infer_video

            full, _ = infer_video(head, npz["features"])
            info["max_abs_diff_vs_forward"] = float(np.abs(np.stack(rows) - full).max())
        log = [
            {
                "class": int(c), "name": names[c], "t_event": float(sec[t]), "t_emit": float(sec[te]),
                "score": float(s), "alarm": bool(s >= params["theta"].get(c, 0.0) if isinstance(params["theta"], dict)
                                                 else s >= params["theta"]),
            }
            for c, t, s, te in events
        ]
        dst.write_text(json.dumps({"patient": p, "info": info, "events": log}))
        report[p] = info
        print(p, info, flush=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
