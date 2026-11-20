#!/usr/bin/env python3
"""
EXP-21 phase 3, step 6: fusion of the encoder heads (arm M) and the long-context models, for the 4 long
events and for the phases. Everything is chosen on the inner val (5 videos) and reported on the 10 held-out
videos of fold 0.

Arms: (a) encoder; (b) long model; (c) per-step confidence-weighted mean, w = conf_a / (conf_a + conf_b);
(c') global weight w chosen on the inner val. Events: confidence = strongest long-event class; phases:
top class. Choices on the inner val (score = long-group mAP averaged over +-1/2/5 s; phases: mean F1@50
after P1): the long model of (b) among a3/sed x seeds 0/1 and the two seed ensembles, the decoder
(nms / caps / caps_chain) of every arm, the weight of (c'), and the final arm. (c) uses the long model of (b).
Per-video Wilcoxon test (two-sided, paired over the 10 test videos) of (c) and (c') against the better of
(a) and (b) (better = higher inner-val score). Short events: encoder heads, decoder chosen on the inner val.

Writes ``<out>/events.json``, ``<out>/phases.json`` (inner-val choices, test numbers, tests) and
``<out>/final_offline.json`` + ``final_phases.npz`` (selected detections / phases, used by the confusion,
correlation and viewer steps). Aggregates only, except per-video values for the paired tests.

    python tools/fuse_focus8.py --out results/p3/fusion [--videos 1] [--boot 1000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F, ev  # noqa: E402

VARIANTS = ("nms", "caps", "caps_chain")
CANDS = P.LONG_RUNS + ["a3_ens", "sed_ens"]
WGRID = [round(float(w), 1) for w in np.arange(0.1, 1.0, 0.1)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def to9(dets5):
    """Long-model class index (1..4) -> index in the 9 focus8 event classes."""
    return {v: [(P.LONG_IDX9[c - 1], t, s) for c, t, s in d] for v, d in dets5.items()}


def pick(cands, fn):
    """(best config, its value) over ``cands`` by ``fn``; the first of equal values wins.

    >>> pick([1, 2, 3], lambda x: -abs(x - 2))
    (2, 0)
    """
    best, bv = None, -np.inf
    for c in cands:
        v = fn(c)
        if v > bv + 1e-12:
            best, bv = c, v
    return best, bv


def run_events(ctx, ctx_t, enc):
    """Selection on ``ctx`` (inner val), report on ``ctx_t``."""
    gts = {ctx.set: ctx.gts(P.LONG_NAMES), ctx_t.set: ctx_t.gts(P.LONG_NAMES)}
    cache = {}

    def long_of(st, cand):
        if (st, cand) not in cache:
            cache[(st, cand)] = P.load_long(cand, st, (ctx if st == ctx.set else ctx_t).videos)
        return cache[(st, cand)]

    ea = {st: {v: P.enc_long_view(z) for v, z in enc[st].items()} for st in enc}

    def sc(c, preds, var, n_boot=0):
        dets = c.detect(preds, P.LONG_NAMES, var)
        return P.eval_events(dets, gts[c.set], P.LONG_NAMES, c.videos, n_boot, 0), dets

    def inner(preds, var):
        return P.score(sc(ctx, preds, var)[0], "long")

    out = {"selection": {}, "test": {}, "tests": {}}
    va, vsc = pick(VARIANTS, lambda v: inner(ea[ctx.set], v))
    out["selection"]["a"] = {"decoder": va, "inner_score": vsc}
    log("events a", va, round(vsc, 4))
    table, best_b = [], None
    for cand in CANDS:
        p = long_of(ctx.set, cand)
        for var in VARIANTS:
            s = inner(p, var)
            table.append({"cand": cand, "decoder": var, "inner_score": s})
            if best_b is None or s > best_b[0] + 1e-12:
                best_b = (s, cand, var)
    out["selection"]["b"] = {"cand": best_b[1], "decoder": best_b[2], "inner_score": best_b[0]}
    out["selection"]["b_candidates_inner_val"] = table
    log("events b", best_b)
    cand = best_b[1]
    pb_val = long_of(ctx.set, cand)
    fc_val = P.fuse_videos(ea[ctx.set], pb_val, "conf", True)
    vc, scc = pick(VARIANTS, lambda v: inner(fc_val, v))
    out["selection"]["c"] = {"cand": cand, "decoder": vc, "inner_score": scc}
    best_g = None
    for w in WGRID:
        fg = P.fuse_videos(ea[ctx.set], pb_val, "global", True, w)
        for var in VARIANTS:
            s = inner(fg, var)
            if best_g is None or s > best_g[0] + 1e-12:
                best_g = (s, w, var)
    out["selection"]["cp"] = {"cand": cand, "w": best_g[1], "decoder": best_g[2], "inner_score": best_g[0]}
    log("events c", vc, round(scc, 4), "c'", best_g)

    pb_t = long_of(ctx_t.set, cand)
    arms = {"a": (ea[ctx_t.set], va), "b": (long_of(ctx_t.set, best_b[1]), best_b[2]),
            "c": (P.fuse_videos(ea[ctx_t.set], pb_t, "conf", True), vc),
            "cp": (P.fuse_videos(ea[ctx_t.set], pb_t, "global", True, best_g[1]), best_g[2])}
    dets_t = {}
    for k, (preds, var) in arms.items():
        res, dets_t[k] = sc(ctx_t, preds, var, BOOT[0])
        out["test"][k] = res
        log("events test", k, var, {d: round(res["group_map"]["long"][str(d)], 3) for d in P.DELTAS})
    scores = {"a": vsc, "b": best_b[0], "c": scc, "cp": best_g[0]}
    better = "a" if scores["a"] >= scores["b"] else "b"
    out["selection"]["better_of_a_b"] = better
    for k in ("c", "cp"):
        out["tests"][f"{k}_vs_{better}"] = {
            str(d): P.wilcoxon(out["test"][k]["per_video_group_ap"]["long"][str(d)],
                               out["test"][better]["per_video_group_ap"]["long"][str(d)]) for d in P.DELTAS}
    final = max(scores, key=lambda k: (scores[k], k == "a"))
    out["selection"]["final_long_arm"] = final
    fpreds = {"a": ea[ctx.set], "b": pb_val, "c": fc_val,
              "cp": P.fuse_videos(ea[ctx.set], pb_val, "global", True, best_g[1])}[final]
    fvar = {"a": va, "b": best_b[2], "c": vc, "cp": best_g[2]}[final]
    return out, to9(sc(ctx, fpreds, fvar)[1]), to9(dets_t[final])


def run_short(ctx, ctx_t, enc):
    gts = {c.set: c.gts(F.EVENT_CLASSES) for c in (ctx, ctx_t)}

    def run(c, var, n_boot=0):
        dets = c.detect(enc[c.set], F.EVENT_CLASSES, var)
        dets = {v: [d for d in x if d[0] in P.SHORT_IDX] for v, x in dets.items()}
        return P.eval_events(dets, gts[c.set], F.EVENT_CLASSES, c.videos, n_boot, 0), dets

    var, s = pick(VARIANTS, lambda v: P.score(run(ctx, v)[0], "short"))
    res, dets_t = run(ctx_t, var, BOOT[0])
    log("short", var, round(s, 4), {d: round(res["group_map"]["short"][str(d)], 3) for d in P.DELTAS})
    return {"decoder": var, "inner_score": s, "test": res}, run(ctx, var)[1], dets_t


def run_phases(ctx, ctx_t, enc):
    out = {"selection": {}, "test": {}, "tests": {}}
    ea = {st: {v: {"second": z["second"], "probs_phases": z["probs_phases"].astype(np.float32)}
               for v, z in enc[st].items()} for st in enc}
    cache = {}

    def long_of(st, cand):
        if (st, cand) not in cache:
            cache[(st, cand)] = P.load_long(cand, st, (ctx if st == ctx.set else ctx_t).videos)
        return cache[(st, cand)]

    def inner(preds):
        return P.eval_phases(preds, ctx)[0]["F1@50_mean_video"]["P1"]

    sa = inner(ea[ctx.set])
    cand, sb = pick(CANDS, lambda c: inner(long_of(ctx.set, c)))
    pb = long_of(ctx.set, cand)
    fc = P.fuse_videos(ea[ctx.set], pb, "conf", False)
    sc_ = inner(fc)
    w, sg = pick(WGRID, lambda w: inner(P.fuse_videos(ea[ctx.set], pb, "global", False, w)))
    out["selection"] = {"a": {"inner_score": sa}, "b": {"cand": cand, "inner_score": sb},
                        "c": {"cand": cand, "inner_score": sc_}, "cp": {"cand": cand, "w": w, "inner_score": sg},
                        "score": "F1@50 (mean over videos) after P1"}
    log("phases inner", out["selection"])
    pbt = long_of(ctx_t.set, cand)
    arms = {"a": ea[ctx_t.set], "b": pbt, "c": P.fuse_videos(ea[ctx_t.set], pbt, "conf", False),
            "cp": P.fuse_videos(ea[ctx_t.set], pbt, "global", False, w)}
    per, decs = {}, {}
    for k, preds in arms.items():
        a, pv, decs[k] = P.eval_phases(preds, ctx_t)
        out["test"][k], per[k] = a, pv
        log("phases test", k, a["F1@50_mean_video"], round(a["row_accuracy"], 3))
    scores = {"a": sa, "b": sb, "c": sc_, "cp": sg}
    better = "a" if sa >= sb else "b"
    out["selection"]["better_of_a_b"] = better
    for k in ("c", "cp"):
        out["tests"][f"{k}_vs_{better}"] = {m: P.wilcoxon(per[k]["F1@50"][m], per[better]["F1@50"][m])
                                           for m in ("P0", "P1")}
    final = max(scores, key=lambda k: (scores[k], k == "a"))
    out["selection"]["final_phase_arm"] = final
    fv = {"a": ea[ctx.set], "b": pb, "c": fc, "cp": P.fuse_videos(ea[ctx.set], pb, "global", False, w)}[final]
    return out, P.eval_phases(fv, ctx)[2], decs[final]


def save_phases(path, dec_val, dec_test):
    z = {}
    for st, dec in (("val", dec_val), ("test", dec_test)):
        for v, d in dec.items():
            for k in ("secs", "gt", "P0", "P1", "probs"):
                z[f"{st}/{v}/{k}"] = np.asarray(d[k])
    np.savez_compressed(path, **z)


BOOT = [1000]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(P.OUT / "fusion"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--videos", type=int, default=0, help="smoke test: first N videos of each split")
    args = ap.parse_args()
    BOOT[0] = args.boot
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv, ct = P.Ctx("inner_val"), P.Ctx("held_out")
    if args.videos:
        for c in (cv, ct):
            c.videos = c.videos[: args.videos]
    enc = {c.set: P.load_dir(P.ENC_DIR, c.videos) for c in (cv, ct)}
    meta = {"enc_dir": str(P.ENC_DIR), "long_dir": str(P.LONG_DIR), "label_dir": ev.LAB4, "splits": str(P.SPLITS),
            "n_videos": {c.set: len(c.videos) for c in (cv, ct)}, "bootstrap": args.boot, "candidates": CANDS}
    e, dv5, dt5 = run_events(cv, ct, enc)
    s, dv_s, dt_s = run_short(cv, ct, enc)
    e["short"], e["meta"] = s, meta
    P.dump_json(e, out / "events.json")
    ph, dec_val, dec_test = run_phases(cv, ct, enc)
    ph["meta"] = meta
    P.dump_json(ph, out / "phases.json")
    dv = {v: dv_s.get(v, []) + dv5.get(v, []) for v in cv.videos}
    dt = {v: dt_s.get(v, []) + dt5.get(v, []) for v in ct.videos}
    P.dump_json({"inner_val": P.to_json_dets(dv), "held_out": P.to_json_dets(dt), "short_decoder": s["decoder"],
                 "long_arm": e["selection"]["final_long_arm"], "phase_arm": ph["selection"]["final_phase_arm"],
                 "classes": F.EVENT_CLASSES}, out / "final_offline.json")
    save_phases(out / "final_phases.npz", dec_val, dec_test)
    (out / ".done").write_text(time.strftime("%F %T"))
    log("done", out)


if __name__ == "__main__":
    main()
