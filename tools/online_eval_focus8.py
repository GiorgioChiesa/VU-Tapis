#!/usr/bin/env python3
"""
EXP-21 phase 3, step 8: online (causal) evaluation on the 10 held-out videos of fold 0.

Uses the cached outputs, which are identical to a streaming run: the encoder clip is centred (16 frames at 4 fps),
so the probability of row t is available 2 s after t (declared delay ``--clip-delay``, 2 s), and the long models
are the causal variants (forward scan, left padded convolutions) on those features.

  * short events: encoder heads, decoded online (``tapis/evaluate/online_decode.py``: a peak at t is emitted at
    t + L, L = 2 s, past suppression window 5 s, no look-ahead beyond L);
  * long events: causal long model (a3 / sed x seeds 0 and 1, and the seed ensembles) decoded the same way;
  * every choice (causal long model, plain or caps+chain decoder, per-class thresholds of the operating point for
    the confusion matrices, long model of the phases) is made on the inner val;
  * metrics: mAP at +-1/2/5 s (group and class, bootstrap CI over videos), mean latency of the true positives
    (emission time + clip delay - ground-truth time) and the declared delay, phases from the last row of each
    second: accuracy and F1@50 for raw argmax (P0) and causal hysteresis (H).
Writes ``<out>/online.json``, ``final_online.json`` (detections with emission times, thresholds) and
``online_phases.npz``. Aggregates only.

    python tools/online_eval_focus8.py --out results/p3/online [--videos 1]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib as P  # noqa: E402
from p3lib import F, ev, pd_, se  # noqa: E402
from tapis.evaluate import online_decode as od  # noqa: E402

FPS = 4
WINDOW_ROWS = 5 * FPS
VARIANTS = ("plain", "caps_chain")
CANDS = P.CAUSAL_RUNS + ["a3_causal_ens", "sed_causal_ens"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def decode(probs, second, classes, variant, pri, delay_s, alpha=0.1):
    """Online emissions ``[(class, time_s, score, emit_time_s)]`` of one video (scores > 0)."""
    caps, chain = (pri["caps"], pri["chain"]) if variant == "caps_chain" else (None, ())
    em = od.decode_online(probs, theta=0.0, window=WINDOW_ROWS, delay=int(round(delay_s * FPS)), caps=caps,
                          chain=chain, alpha=alpha, classes=classes)
    return [(c, float(second[t]), float(s), float(second[te])) for c, t, s, te in em if s > 0]


def drop_ign(em, ign):
    keep = se.drop_ignored([(c, t, s) for c, t, s, _ in em], ign)
    ks = {(c, t, s) for c, t, s in keep}
    return [e for e in em if (e[0], e[1], e[2]) in ks]


def latency(em_by_video, gts, delta, clip_delay):
    """Mean / std / n of ``emit time + clip delay - ground-truth time`` over the true positives (any class)."""
    lat = []
    for v, g in gts.items():
        for c in {k for k, _ in g}:
            e = [x for x in em_by_video[v] if x[0] == c]
            gt = [t for k, t in g if k == c]
            for i, j in P.match_with_gt([x[1] for x in e], [x[2] for x in e], gt, delta):
                lat.append(e[i][3] + clip_delay - gt[j])
    return {"mean": float(np.mean(lat)) if lat else float("nan"), "std": float(np.std(lat)) if lat else float("nan"),
            "n_tp": len(lat)}


def last_row_per_second(probs, second):
    """Probabilities of the last row of every second (the only causal 1 Hz summary), and the second index."""
    s = np.floor(np.asarray(second) + 1e-6).astype(int)
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    return probs[last], s[last]


def phases_online(preds, ctx, margin=0.1):
    """Online phase metrics (P0 = argmax, H = causal hysteresis) and the decoded labels per video."""
    C = len(F.PHASE_IDS)
    f1, acc, dec = {"P0": [], "H": []}, {"P0": [], "H": []}, {}
    for v in ctx.videos:
        p1, secs = last_row_per_second(preds[v]["probs_phases"], preds[v]["second"])
        g = ev.row_phase(ctx.labs[v], secs)
        d = {"P0": p1.argmax(1), "H": pd_.hysteresis(p1, margin)}
        for k, x in d.items():
            m = pd_.video_metrics(x, g, C)
            f1[k].append(float(m["F1@50"]))
            acc[k].append(float(m["acc"]))
        dec[v] = {"secs": secs, "gt": g, "P0": d["P0"], "P1": d["H"], "probs": p1}
    return {"F1@50_mean_video": {k: float(np.mean(x)) for k, x in f1.items()},
            "acc_1hz_mean_video": {k: float(np.mean(x)) for k, x in acc.items()}}, dec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(P.OUT / "online"))
    ap.add_argument("--clip-delay", type=float, default=2.0)
    ap.add_argument("--decoder-delay", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--videos", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    cv, ct = P.Ctx("inner_val"), P.Ctx("held_out")
    if a.videos:
        for c in (cv, ct):
            c.videos = c.videos[: a.videos]
    enc = {c.set: P.load_dir(P.ENC_DIR, c.videos) for c in (cv, ct)}
    names9, names5 = F.EVENT_CLASSES, P.LONG_NAMES
    gts9 = {c.set: c.gts(names9) for c in (cv, ct)}
    gts5 = {c.set: c.gts(names5) for c in (cv, ct)}
    cache = {}

    def long_of(st, cand):
        if (st, cand) not in cache:
            cache[(st, cand)] = P.load_long(cand, st, (cv if st == "inner_val" else ct).videos)
        return cache[(st, cand)]

    def dec_short(c, var):
        pri = c.priors(names9)
        return {v: drop_ign(decode(enc[c.set][v]["probs_steps"], enc[c.set][v]["second"], P.SHORT_IDX, var, pri,
                                   a.decoder_delay), c.ign[v]) for v in c.videos}

    def dec_long(c, cand, var):
        pri, p = c.priors(names5), long_of(c.set, cand)
        em5 = {v: drop_ign(decode(p[v]["probs_steps"], p[v]["second"], [1, 2, 3, 4], var, pri, a.decoder_delay),
                           c.ign[v]) for v in c.videos}
        return {v: [(P.LONG_IDX9[c_ - 1], t, s, te) for c_, t, s, te in e] for v, e in em5.items()}

    def spot(c, em, n_boot=0):
        dets = {v: [(c_, t, s) for c_, t, s, _ in e] for v, e in em.items()}
        return P.eval_events(dets, gts9[c.set], names9, c.videos, n_boot, 0)

    res = {"selection": {}, "test": {}, "declared": {"clip_delay_s": a.clip_delay, "decoder_delay_s": a.decoder_delay,
                                                       "window_s": WINDOW_ROWS / FPS}}
    vs, ss = max(((v, P.score(spot(cv, dec_short(cv, v)), "short")) for v in VARIANTS), key=lambda x: x[1])
    best = None
    for cand in CANDS:
        for var in VARIANTS:
            s = P.score(spot(cv, dec_long(cv, cand, var)), "long")
            if best is None or s > best[0] + 1e-12:
                best = (s, cand, var)
    res["selection"].update({"short": {"decoder": vs, "inner_score": ss},
                             "long": {"cand": best[1], "decoder": best[2], "inner_score": best[0]}})
    log("selection", res["selection"])
    em_val = {v: dec_short(cv, vs)[v] + dec_long(cv, best[1], best[2])[v] for v in cv.videos}
    em_t = {v: dec_short(ct, vs)[v] + dec_long(ct, best[1], best[2])[v] for v in ct.videos}
    res["test"]["events"] = spot(ct, em_t, a.boot)
    res["test"]["latency"] = {str(d): latency(em_t, gts9["held_out"], d, a.clip_delay) for d in P.DELTAS}
    res["test"]["latency_inner_val"] = {str(d): latency(em_val, gts9["inner_val"], d, a.clip_delay) for d in P.DELTAS}
    log("test group mAP", res["test"]["events"]["group_map"], "latency", res["test"]["latency"]["2"])
    # operating points on the inner val (for the online confusion matrices)
    dets_val = {v: [(c, t, s) for c, t, s, _ in e] for v, e in em_val.items()}
    th = P.best_thresholds(dets_val, gts9["inner_val"], list(range(1, len(names9))), 2.0)
    res["thresholds"] = {names9[c]: t for c, t in th.items()}
    # phases
    best_p = max(((c, phases_online(long_of("inner_val", c), cv)[0]["F1@50_mean_video"]["H"]) for c in CANDS),
                 key=lambda x: x[1])
    res["selection"]["phases"] = {"cand": best_p[0], "inner_score_F1@50_H": best_p[1]}
    pa, dec_t = phases_online(long_of("held_out", best_p[0]), ct)
    _, dec_v = phases_online(long_of("inner_val", best_p[0]), cv)
    res["test"]["phases"] = pa
    log("phases test", pa)
    res["meta"] = {"long_dir": str(P.LONG_DIR), "enc_dir": str(P.ENC_DIR), "n_videos": {c.set: len(c.videos) for c in (cv, ct)}}
    P.dump_json(res, out / "online.json")
    P.dump_json({"inner_val": {v: [list(e) for e in x] for v, x in em_val.items()},
                 "held_out": {v: [list(e) for e in x] for v, x in em_t.items()}, "thresholds": res["thresholds"],
                 "classes": names9}, out / "final_online.json")
    z = {}
    for st, dec in (("val", dec_v), ("test", dec_t)):
        for v, d in dec.items():
            for k in ("secs", "gt", "P0", "P1", "probs"):
                z[f"{st}/{v}/{k}"] = np.asarray(d[k])
    np.savez_compressed(out / "online_phases.npz", **z)
    (out / ".done").write_text(time.strftime("%F %T"))


if __name__ == "__main__":
    main()
