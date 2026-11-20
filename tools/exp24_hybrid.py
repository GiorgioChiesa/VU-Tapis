#!/usr/bin/env python3
"""
EXP-24, one fold: hybrid of the reference S MS-TCN (short events) and L-A3 on the S features (long events, phases).

Detections are merged per group: short classes from reference_s<k> (decoder nms), long classes from L-A3 with the seed /
seed ensemble and decoder (nms, caps_chain) chosen on the fold's inner val (long-group mAP over +-1/2/5 s). The hybrid is built
per reference seed k and its metrics are averaged over k, as the reference's. Bootstrap (1000, per video, same resamples for all
models) gives paired CIs. Writes results/exp24/fold<i>/{fold.json, boot.npz, hybrid_dets_s0.json}. Aggregates and event times only.

    python tools/exp24_hybrid.py --fold 0 [--online]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp22_report as R22  # noqa: E402
import p3lib as P  # noqa: E402
from p3lib import F, ev  # noqa: E402

NAMES = F.EVENT_CLASSES
CLASSES = list(range(1, len(NAMES)))
SHORT_C = [c for c in CLASSES if NAMES[c] in F.SHORT]
LONG_C = [c for c in CLASSES if NAMES[c] in F.LONG]
KEYS = [NAMES[c] for c in CLASSES] + ["short", "long", "mean8"]
DEC = ("nms", "caps_chain")
OUT = P.REPO / "results/exp24"


def pick(dets, cls):
    return {v: [x for x in d if x[0] in cls] for v, d in dets.items()}


def merge(a, b):
    return {v: sorted(a[v] + b[v], key=lambda x: (x[1], x[0])) for v in a}


def vec(ap):
    """ap[d][c] -> [len(KEYS), 3]: 8 class APs, short / long group mAP, mean of the 8 class APs."""
    out = np.full((len(KEYS), len(P.DELTAS)), np.nan)
    for j, d in enumerate(P.DELTAS):
        a = [ap[d][c] for c in CLASSES]
        out[:len(CLASSES), j] = a
        out[-3, j] = np.nanmean([ap[d][c] for c in SHORT_C])
        out[-2, j] = np.nanmean([ap[d][c] for c in LONG_C])
        out[-1, j] = np.nanmean(a)
    return out


def metrics(dets_list, gts, videos, draws):
    """Mean over the seeds of: point [K,3], bootstrap [B,K,3], and the per-video AP of short / long / all 8 classes."""
    pts, boots, pvs = [], [], []
    for dets in dets_list:
        m, n = ev.matches(dets, gts, CLASSES)
        pts.append(vec(ev.ap_table(m, n, videos, CLASSES)))
        boots.append(np.stack([vec(ev.ap_table(m, n, list(dr), CLASSES)) for dr in draws]))
        pvs.append({g: P.per_video_group_ap(m, n, videos, CLASSES, cl)
                    for g, cl in (("short", SHORT_C), ("long", LONG_C), ("mean8", CLASSES))})
    pv = {g: {str(d): {v: float(np.nanmean([p[g][d][v] for p in pvs])) for v in videos} for d in P.DELTAS} for g in pvs[0]}
    return np.mean(pts, 0), np.mean(boots, 0), pv


def l3_runs(R, ctx, stem="a3"):
    runs = []
    for s in (0, 1):
        preds, names = R22.load(R / f"{stem}_s{s}", ctx)
        assert names == NAMES
        runs.append(preds)
    return {"s0": runs[0], "s1": runs[1], "ens": R22.ensemble(runs)}


def phase_summary(agg_list, pv_list):
    keys = {"row_accuracy": lambda a: a["row_accuracy"], "row_macro_f1": lambda a: a["row_macro_f1"],
            "F1@50_P0": lambda a: a["F1@50_mean_video"]["P0"], "F1@50_P1": lambda a: a["F1@50_mean_video"]["P1"],
            "acc1hz_P0": lambda a: a["acc_1hz_mean_video"]["P0"], "acc1hz_P1": lambda a: a["acc_1hz_mean_video"]["P1"]}
    vids = list(pv_list[0]["F1@50"]["P0"])
    pv = {f"{m}_{k}": {v: float(np.mean([p[m][k][v] for p in pv_list])) for v in vids}
          for m in ("F1@50", "acc_1hz") for k in ("P0", "P1")}
    return {k: float(np.mean([f(a) for a in agg_list])) for k, f in keys.items()}, pv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--online", action="store_true")
    a = ap.parse_args()
    f = a.fold
    R = R22.EXP22 / f"fold{f}"
    out = OUT / f"fold{f}"
    out.mkdir(parents=True, exist_ok=True)
    P.SPLITS = R / "splits.json"
    cv, ct = P.Ctx("inner_val"), P.Ctx("held_out")

    # ---- choices on the inner val only
    gcv = cv.gts(NAMES)
    cands = l3_runs(R, cv)
    sel, phs_sel = {}, {}
    for c, preds in cands.items():
        for d in DEC:
            sel[f"{c}|{d}"] = P.score(ev.spotting(cv.detect(preds, NAMES, d), gcv, NAMES, 0, 0), "long")
        phs_sel[c] = P.eval_phases(preds, cv)[0]["F1@50_mean_video"]["P1"]
    best = max(sel, key=sel.get)
    cand, dec = best.split("|")
    pcand = max(phs_sel, key=phs_sel.get)
    print(f"fold {f}: long events from L-A3 {cand} / {dec} (inner-val long score {sel[best]:.3f}); phases from {pcand}", flush=True)

    # ---- held-out
    gts = ct.gts(NAMES)
    lt = l3_runs(R, ct)
    d_sel = ct.detect(lt[cand], NAMES, dec)
    refs = []
    for k in (0, 1):
        assert json.loads((R / f"reference_s{k}/meta.json").read_text())["event_classes"] == NAMES
        refs.append(P.load_dir(R / f"reference_s{k}", ct.videos))
    d_ref = [ct.detect(r, NAMES, "nms") for r in refs]
    d_hyb = [merge(pick(d, SHORT_C), pick(d_sel, LONG_C)) for d in d_ref]
    models = {"hybrid": d_hyb, "reference": d_ref, "L-A3 (EXP-22 rule)": [ct.detect(lt[s], NAMES, "nms") for s in ("s0", "s1")],
              "L-A3 (selected)": [d_sel]}
    rng = np.random.default_rng(0)
    draws = [[str(x) for x in rng.choice(ct.videos, len(ct.videos), replace=True)] for _ in range(a.boot)]
    res, boots = {}, {}
    for name, dl in models.items():
        pt, bt, pv = metrics(dl, gts, ct.videos, draws)
        boots[name] = bt
        ci = np.nanpercentile(bt, [2.5, 97.5], axis=0)
        res[name] = {"point": pt.tolist(), "ci95": ci.tolist(), "per_video": pv}
        print("evaluated", name, flush=True)
    dif = boots["hybrid"] - boots["reference"]
    res["hybrid_minus_reference"] = {"point": (np.array(res["hybrid"]["point"]) - np.array(res["reference"]["point"])).tolist(),
                                     "ci95": np.nanpercentile(dif, [2.5, 97.5], axis=0).tolist()}
    short_diff = float(np.abs(np.array(res["hybrid"]["point"])[KEYS.index("short")]
                              - np.array(res["reference"]["point"])[KEYS.index("short")]).max())

    # ---- phases (hybrid = L-A3 selected phase candidate)
    ph = {}
    ref_ph = [P.eval_phases(r, ct)[:2] for r in refs]
    ph["reference"] = phase_summary([x[0] for x in ref_ph], [x[1] for x in ref_ph])
    l3p = {s: P.eval_phases(lt[s], ct)[:2] for s in ("s0", "s1", "ens")}
    ph["hybrid"] = phase_summary([l3p[pcand][0]], [l3p[pcand][1]])
    ph["L-A3 (EXP-22 rule)"] = phase_summary([l3p[s][0] for s in ("s0", "s1")], [l3p[s][1] for s in ("s0", "s1")])

    report = {"fold": f, "keys": KEYS, "deltas": list(P.DELTAS), "selection": {"inner_val_long_score": sel, "chosen": best,
              "phase_inner_val_F1@50_P1": phs_sel, "phase_cand": pcand}, "short_diff_max": short_diff, "events": res,
              "phases": ph, "n_boot": a.boot, "videos": ct.videos, "label_dir": str(ct.label_dir), "splits": str(P.SPLITS)}

    if a.online:
        import online_eval_focus8 as O

        cc = {"inner_val": l3_runs(R, cv, "a3_causal"), "held_out": l3_runs(R, ct, "a3_causal")}
        pri_c = {c.set: c.priors(NAMES) for c in (cv, ct)}

        def emis(c, preds, var):
            return {v: O.drop_ign(O.decode(preds[v]["probs_steps"], preds[v]["second"], CLASSES, var, pri_c[c.set], 2.0), c.ign[v])
                    for v in c.videos}

        def to_dets(em, cls):
            return {v: [(k, t, s) for k, t, s, _ in e if k in cls] for v, e in em.items()}

        osel = {}
        for c, preds in cc["inner_val"].items():
            for var in O.VARIANTS:
                osel[f"{c}|{var}"] = P.score(ev.spotting(to_dets(emis(cv, preds, var), LONG_C), gcv, NAMES, 0, 0), "long")
        ob = max(osel, key=osel.get)
        oc, ov = ob.split("|")
        em = emis(ct, cc["held_out"][oc], ov)
        d_on = to_dets(em, LONG_C)
        om = {"hybrid online (causal L-A3 long + reference short, offline)": [merge(pick(d, SHORT_C), d_on) for d in d_ref],
              "causal L-A3 alone (selected)": [to_dets(em, CLASSES)]}
        glong = {v: [(k, t) for k, t in g if k in LONG_C] for v, g in gts.items()}
        lat = {str(d): O.latency({v: [e for e in x if e[0] in LONG_C] for v, x in em.items()}, glong, d, 2.0) for d in P.DELTAS}
        ores = {}
        for name, dl in om.items():
            pt, bt, _ = metrics(dl, gts, ct.videos, draws)
            ores[name] = {"point": pt.tolist(), "ci95": np.nanpercentile(bt, [2.5, 97.5], axis=0).tolist()}
        report["online"] = {"selection_inner_val_long_score": osel, "chosen": ob, "results": ores, "latency_long_tp": lat,
                            "note": "reference short events are offline detections with the declared 2 s delay"}

    P.dump_json(report, out / "fold.json")
    np.savez_compressed(out / "boot.npz", **{k.replace(" ", "_"): v.astype(np.float32) for k, v in boots.items()})
    P.dump_json(P.to_json_dets(d_hyb[0]), out / "hybrid_dets_s0.json")
    g = lambda n, k: [round(x, 3) for x in np.array(res[n]["point"])[KEYS.index(k)]]
    print(f"fold {f} @1/2/5 short: hybrid {g('hybrid','short')} ref {g('reference','short')}; long: hybrid {g('hybrid','long')} "
          f"ref {g('reference','long')} L-A3 {g('L-A3 (EXP-22 rule)','long')}; short_diff_max {short_diff:.2e}")


if __name__ == "__main__":
    main()
