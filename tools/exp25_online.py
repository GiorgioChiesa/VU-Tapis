#!/usr/bin/env python3
"""
EXP-25, one fold: fully online hybrid = causal S MS-TCN (short events) + causal L-A3 (long events, phases), all decoded online
(tools/online_eval_focus8.py: decoder delay 2 s, clip delay 2 s, window 5 s; decoders plain / caps_chain).

Choices on the fold's inner val only: short part = (seed 0 | seed 1 | seed ensemble) x decoder by the short-group mAP averaged
over +-1/2/5 s; long part = same for the causal L-A3 with the long-group mAP; phases = causal L-A3 candidate with the best
inner-val F1@50 of the causal hysteresis. Held-out metrics with per-video bootstrap (1000, seed 0, same resamples as EXP-24) and
the comparison with the EXP-24 offline hybrid / offline reference (results/exp24/fold<i>/{fold.json,boot.npz}).
Writes results/exp25/fold<i>/{fold.json, boot.npz, online_hybrid_dets.json}. Aggregates and event times only.

    python tools/exp25_online.py --fold 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp22_report as R22  # noqa: E402
import exp24_hybrid as H  # noqa: E402
import online_eval_focus8 as O  # noqa: E402
import p3lib as P  # noqa: E402
from p3lib import F, ev, pd_  # noqa: E402

NAMES, CLASSES, SHORT_C, LONG_C, KEYS = H.NAMES, H.CLASSES, H.SHORT_C, H.LONG_C, H.KEYS
OUT = P.REPO / "results/exp25"
CANDS = ("s0", "s1", "ens")


def runs_of(dirs, ctx):
    rs = []
    for d in dirs:
        preds, names = R22.load(d, ctx)
        assert names == NAMES, (d, names)
        rs.append(preds)
    return {"s0": rs[0], "s1": rs[1], "ens": R22.ensemble(rs)}


def latency_list(em_by_video, gts, delta, clip_delay=2.0):
    lat = []
    for v, g in gts.items():
        for c in {k for k, _ in g}:
            e = [x for x in em_by_video[v] if x[0] == c]
            gt = [t for k, t in g if k == c]
            for i, j in P.match_with_gt([x[1] for x in e], [x[2] for x in e], gt, delta):
                lat.append(e[i][3] + clip_delay - gt[j])
    return lat


def phase_per_video(preds, ctx):
    """Online phases: aggregate from O.phases_online plus per-video F1@50 for P0 and H."""
    agg, dec = O.phases_online(preds, ctx)
    C = len(F.PHASE_IDS)
    pv = {k: {v: float(pd_.video_metrics(d[k2], d["gt"], C)["F1@50"]) for v, d in dec.items()} for k, k2 in (("P0", "P0"), ("H", "P1"))}
    return agg, pv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    f = a.fold
    R, X = R22.EXP22 / f"fold{f}", OUT / f"fold{f}"
    X.mkdir(parents=True, exist_ok=True)
    P.SPLITS = R / "splits.json"
    cv, ct = P.Ctx("inner_val"), P.Ctx("held_out")
    gcv, gct = cv.gts(NAMES), ct.gts(NAMES)
    short_dirs = [X / f"short_causal_s{s}" for s in (0, 1)]
    long_dirs = [(R if f == 0 else X) / f"a3_causal_s{s}" for s in (0, 1)]
    for d in short_dirs + long_dirs:
        assert (d / "held_out").exists() and (d / "inner_val").exists(), f"missing run {d}"
    S = {"inner_val": runs_of(short_dirs, cv), "held_out": runs_of(short_dirs, ct)}
    L = {"inner_val": runs_of(long_dirs, cv), "held_out": runs_of(long_dirs, ct)}
    pri = {c.set: c.priors(NAMES) for c in (cv, ct)}

    def emis(c, preds, var):
        return {v: O.drop_ign(O.decode(preds[v]["probs_steps"], preds[v]["second"], CLASSES, var, pri[c.set], 2.0), c.ign[v])
                for v in c.videos}

    def dets_of(em, cls):
        return {v: [(k, t, s) for k, t, s, _ in e if k in cls] for v, e in em.items()}

    # ---- choices on the inner val only
    sel_s, sel_l, sel_n, sel_p = {}, {}, {}, {}
    for c in CANDS:
        for var in O.VARIANTS:
            sel_s[f"{c}|{var}"] = P.score(ev.spotting(dets_of(emis(cv, S["inner_val"][c], var), SHORT_C), gcv, NAMES, 0, 0), "short")
            sel_l[f"{c}|{var}"] = P.score(ev.spotting(dets_of(emis(cv, L["inner_val"][c], var), LONG_C), gcv, NAMES, 0, 0), "long")
        sel_n[c] = P.score(ev.spotting(cv.detect(S["inner_val"][c], NAMES, "nms"), gcv, NAMES, 0, 0), "short")
        sel_p[c] = phase_per_video(L["inner_val"][c], cv)[0]["F1@50_mean_video"]["H"]
    bs, bl, bn, bp = max(sel_s, key=sel_s.get), max(sel_l, key=sel_l.get), max(sel_n, key=sel_n.get), max(sel_p, key=sel_p.get)
    print(f"fold {f}: short {bs} ({sel_s[bs]:.3f}); long {bl} ({sel_l[bl]:.3f}); phases {bp} ({sel_p[bp]:.1f}); short nms {bn}", flush=True)

    # ---- held-out
    cs, vs = bs.split("|")
    cl, vl = bl.split("|")
    em_s = emis(ct, S["held_out"][cs], vs)
    em_l = emis(ct, L["held_out"][cl], vl)
    em_h = {v: [e for e in em_s[v] if e[0] in SHORT_C] + [e for e in em_l[v] if e[0] in LONG_C] for v in ct.videos}
    d_hyb = {v: sorted([(k, t, s) for k, t, s, _ in e], key=lambda x: (x[1], x[0])) for v, e in em_h.items()}
    d_short = dets_of(em_s, SHORT_C)
    d_short_nms = {v: [x for x in d if x[0] in SHORT_C] for v, d in ct.detect(S["held_out"][bn], NAMES, "nms").items()}
    rng = np.random.default_rng(0)
    draws = [[str(x) for x in rng.choice(ct.videos, len(ct.videos), replace=True)] for _ in range(a.boot)]
    models = {"online hybrid": d_hyb, "causal short MS-TCN alone (online decoding)": d_short,
              "causal short MS-TCN alone (nms decoding, offline decoder)": d_short_nms}
    res, boots = {}, {}
    for name, d in models.items():
        pt, bt, pv = H.metrics([d], gct, ct.videos, draws)
        boots[name] = bt
        res[name] = {"point": pt.tolist(), "ci95": np.nanpercentile(bt, [2.5, 97.5], axis=0).tolist(), "per_video": pv}
        print("evaluated", name, flush=True)
    # comparison with the offline models of EXP-24 (same bootstrap draws: same rng, same videos)
    f24 = json.loads((P.REPO / f"results/exp24/fold{f}/fold.json").read_text())
    b24 = dict(np.load(P.REPO / f"results/exp24/fold{f}/boot.npz"))
    assert f24["videos"] == ct.videos
    comp = {}
    for off in ("hybrid", "reference"):
        dif = boots["online hybrid"] - b24[off]
        comp[f"online hybrid - offline {off}"] = {
            "point": (np.array(res["online hybrid"]["point"]) - np.array(f24["events"][off]["point"])).tolist(),
            "ci95": np.nanpercentile(dif, [2.5, 97.5], axis=0).tolist()}
    ref_short = np.array(f24["events"]["reference"]["point"])[KEYS.index("short")]
    dif = boots["causal short MS-TCN alone (online decoding)"][:, KEYS.index("short")] - b24["reference"][:, KEYS.index("short")]
    comp["causal short alone - offline reference (short group)"] = {
        "point": (np.array(res["causal short MS-TCN alone (online decoding)"]["point"])[KEYS.index("short")] - ref_short).tolist(),
        "ci95": np.nanpercentile(dif, [2.5, 97.5], axis=0).tolist()}

    # ---- latency (true positives of the 8 classes; hybrid, and per group)
    lat = {}
    for d in P.DELTAS:
        for nm, cls in (("hybrid", CLASSES), ("short", SHORT_C), ("long", LONG_C)):
            gg = {v: [(k, t) for k, t in g if k in cls] for v, g in gct.items()}
            ee = {v: [e for e in x if e[0] in cls] for v, x in em_h.items()}
            lat[f"{nm}|{d}"] = latency_list(ee, gg, d)

    # ---- phases (causal L-A3)
    agg_t, pv_t = phase_per_video(L["held_out"][bp], ct)
    f24ph = {m: f24["phases"][m] for m in ("hybrid", "reference")}

    report = {"fold": f, "keys": KEYS, "deltas": list(P.DELTAS), "videos": ct.videos, "n_boot": a.boot,
              "selection": {"short_inner_val": sel_s, "short_chosen": bs, "long_inner_val": sel_l, "long_chosen": bl,
                            "short_nms_inner_val": sel_n, "short_nms_chosen": bn, "phase_inner_val_F1@50_H": sel_p, "phase_cand": bp},
              "events": res, "comparison": comp, "latency_tp": lat,
              "phases": {"online": {"agg": agg_t, "per_video": pv_t}, "offline_hybrid": f24ph["hybrid"], "offline_reference": f24ph["reference"]},
              "label_dir": str(ct.label_dir), "splits": str(P.SPLITS), "short_runs": [str(d) for d in short_dirs],
              "long_runs": [str(d) for d in long_dirs],
              "declared": {"clip_delay_s": 2.0, "decoder_delay_s": 2.0, "window_s": O.WINDOW_ROWS / O.FPS}}
    P.dump_json(report, X / "fold.json")
    np.savez_compressed(X / "boot.npz", **{k.replace(" ", "_"): v.astype(np.float32) for k, v in boots.items()})
    P.dump_json(P.to_json_dets(d_hyb), X / "online_hybrid_dets.json")
    g = lambda n, k: [round(x, 3) for x in np.array(res[n]["point"])[KEYS.index(k)]]
    print(f"fold {f} @1/2/5 mean8 online hybrid {g('online hybrid','mean8')} short {g('online hybrid','short')} long {g('online hybrid','long')}; "
          f"latency @2 {np.mean(lat['hybrid|2']):.2f} s (n {len(lat['hybrid|2'])}); phases H {agg_t['F1@50_mean_video']['H']:.1f}")


if __name__ == "__main__":
    main()
