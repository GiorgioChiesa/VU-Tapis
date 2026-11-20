#!/usr/bin/env python3
"""
EXP-25a, one fold and one look-ahead F: online hybrid = short MS-TCN with look-ahead F s (results/exp25a/fold<i>/short_la_s<seed>/F<F>)
+ strictly causal L-A3 long events (EXP-25 runs and EXP-25 inner-val choice, unchanged), decoded online (decoder delay 2 s, clip delay 2 s).

Short part: (seed 0 | seed 1 | seed ensemble) x decoder (plain, caps_chain) chosen on the fold's inner val by the short-group mAP averaged over
+-1/2/5 s, as EXP-25. Latency of a true positive of the short part = emission time + F + clip delay (2 s) - ground truth (the probability
of row t is available F s after t); long part as EXP-25. Bootstrap 1000 per video, seed 0, same draws as EXP-24 / EXP-25.
Writes results/exp25a/fold<i>/eval_F<F>/{fold.json, boot.npz, online_hybrid_dets.json}. Aggregates and event times only.

    python tools/exp25a_eval.py --fold 0 --F 6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp22_report as R22  # noqa: E402
import exp24_hybrid as H  # noqa: E402
import exp25_online as E25  # noqa: E402
import online_eval_focus8 as O  # noqa: E402
import p3lib as P  # noqa: E402
from p3lib import ev  # noqa: E402

NAMES, CLASSES, SHORT_C, LONG_C, KEYS = H.NAMES, H.CLASSES, H.SHORT_C, H.LONG_C, H.KEYS
OUT = P.REPO / "results/exp25a"
O25 = P.REPO / "results/exp25"
CANDS = ("s0", "s1", "ens")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--F", type=float, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    f, Fs = a.fold, a.F
    R = R22.EXP22 / f"fold{f}"
    X = OUT / f"fold{f}" / f"eval_F{Fs:g}"
    X.mkdir(parents=True, exist_ok=True)
    P.SPLITS = R / "splits.json"
    cv, ct = P.Ctx("inner_val"), P.Ctx("held_out")
    gcv, gct = cv.gts(NAMES), ct.gts(NAMES)
    short_dirs = [OUT / f"fold{f}/short_la_s{s}/F{Fs:g}" for s in (0, 1)]
    long_dirs = [(R if f == 0 else O25 / f"fold{f}") / f"a3_causal_s{s}" for s in (0, 1)]
    for d in short_dirs + long_dirs:
        assert (d / "held_out").exists() and (d / "inner_val").exists(), f"missing run {d}"
    S = {"inner_val": E25.runs_of(short_dirs, cv), "held_out": E25.runs_of(short_dirs, ct)}
    L = {"held_out": E25.runs_of(long_dirs, ct)}
    f25 = json.loads((O25 / f"fold{f}/fold.json").read_text())
    cl, vl = f25["selection"]["long_chosen"].split("|")     # EXP-25 inner-val choice of the long part (independent of the short part)
    pri = {c.set: c.priors(NAMES) for c in (cv, ct)}

    def emis(c, preds, var):
        return {v: O.drop_ign(O.decode(preds[v]["probs_steps"], preds[v]["second"], CLASSES, var, pri[c.set], 2.0), c.ign[v])
                for v in c.videos}

    def dets_of(em, cls):
        return {v: [(k, t, s) for k, t, s, _ in e if k in cls] for v, e in em.items()}

    sel_s = {}
    for c in CANDS:
        for var in O.VARIANTS:
            sel_s[f"{c}|{var}"] = P.score(ev.spotting(dets_of(emis(cv, S["inner_val"][c], var), SHORT_C), gcv, NAMES, 0, 0), "short")
    bs = max(sel_s, key=sel_s.get)
    print(f"fold {f} F {Fs:g}: short {bs} ({sel_s[bs]:.3f}); long (EXP-25) {cl}|{vl}", flush=True)

    cs, vs = bs.split("|")
    em_s = emis(ct, S["held_out"][cs], vs)
    em_l = emis(ct, L["held_out"][cl], vl)
    em_h = {v: [e for e in em_s[v] if e[0] in SHORT_C] + [e for e in em_l[v] if e[0] in LONG_C] for v in ct.videos}
    d_hyb = {v: sorted([(k, t, s) for k, t, s, _ in e], key=lambda x: (x[1], x[0])) for v, e in em_h.items()}
    d_short = dets_of(em_s, SHORT_C)
    rng = np.random.default_rng(0)
    draws = [[str(x) for x in rng.choice(ct.videos, len(ct.videos), replace=True)] for _ in range(a.boot)]
    models = {"online hybrid": d_hyb, "short part alone (online decoding)": d_short}
    res, boots = {}, {}
    for name, d in models.items():
        pt, bt, pv = H.metrics([d], gct, ct.videos, draws)
        boots[name] = bt
        res[name] = {"point": pt.tolist(), "ci95": np.nanpercentile(bt, [2.5, 97.5], axis=0).tolist(), "per_video": pv}
    f24 = json.loads((P.REPO / f"results/exp24/fold{f}/fold.json").read_text())
    b24 = dict(np.load(P.REPO / f"results/exp24/fold{f}/boot.npz"))
    b25 = dict(np.load(O25 / f"fold{f}/boot.npz"))
    assert f24["videos"] == ct.videos and f25["videos"] == ct.videos
    comp = {}
    for nm, ref_boot, ref_pt in (("offline hybrid", b24["hybrid"], f24["events"]["hybrid"]["point"]),
                                 ("EXP-25 (F=0)", b25["online_hybrid"], f25["events"]["online hybrid"]["point"])):
        dif = boots["online hybrid"] - ref_boot
        comp[f"online hybrid F={Fs:g} - {nm}"] = {"point": (np.array(res["online hybrid"]["point"]) - np.array(ref_pt)).tolist(),
                                                   "ci95": np.nanpercentile(dif, [2.5, 97.5], axis=0).tolist()}
    # latency: short part + F (probability of row t is available F s after t), long part unchanged
    em_lat = {v: [(k, t, s, te + Fs) for k, t, s, te in em_s[v] if k in SHORT_C] + [e for e in em_l[v] if e[0] in LONG_C] for v in ct.videos}
    lat = {}
    for d in P.DELTAS:
        for nm, cls in (("hybrid", CLASSES), ("short", SHORT_C), ("long", LONG_C)):
            gg = {v: [(k, t) for k, t in g if k in cls] for v, g in gct.items()}
            ee = {v: [e for e in x if e[0] in cls] for v, x in em_lat.items()}
            lat[f"{nm}|{d}"] = E25.latency_list(ee, gg, d)
    report = {"fold": f, "F": Fs, "keys": KEYS, "deltas": list(P.DELTAS), "videos": ct.videos, "n_boot": a.boot,
              "selection": {"short_inner_val": sel_s, "short_chosen": bs, "short_inner_val_score": sel_s[bs],
                            "long_chosen_exp25": f"{cl}|{vl}"},
              "events": res, "comparison": comp, "latency_tp": lat, "short_runs": [str(d) for d in short_dirs],
              "long_runs": [str(d) for d in long_dirs], "label_dir": str(ct.label_dir), "splits": str(P.SPLITS),
              "declared": {"clip_delay_s": 2.0, "decoder_delay_s": 2.0, "lookahead_s": Fs, "window_s": O.WINDOW_ROWS / O.FPS}}
    P.dump_json(report, X / "fold.json")
    np.savez_compressed(X / "boot.npz", **{k.replace(" ", "_"): v.astype(np.float32) for k, v in boots.items()})
    P.dump_json(P.to_json_dets(d_hyb), X / "online_hybrid_dets.json")
    g = lambda n, k: [round(x, 3) for x in np.array(res[n]["point"])[KEYS.index(k)]]
    print(f"fold {f} F {Fs:g} @1/2/5 mean8 {g('online hybrid','mean8')} short {g('online hybrid','short')} long {g('online hybrid','long')}; "
          f"latency @2 {np.mean(lat['hybrid|2']):.2f} s (n {len(lat['hybrid|2'])})")


if __name__ == "__main__":
    main()
