#!/usr/bin/env python3
"""
Experiment 2 table: does information on phases and atomic actions help Orsi event spotting?

Runs are ``outputs/orsi/EXP2/<arm>/seed<s>/fold<i>/val`` (run_files/exp2_multisource.sh) plus the
reference arms 2.1 = H* of Experiment 1 (``outputs/orsi/HYBRID/G2_dilate3``) and T_base = its two-stream
version (``outputs/orsi/HYBRID/G2_dilate3-two``). For every arm: mean +/- std over the (seed, fold) runs of
the spotting metrics with NMS only and with the caps + order chain decoding (``cc``), paired Wilcoxon
against the reference (see ``reference``), and AP@5 per family of events.

Families group the 32 Orsi event classes by what happens at the onset (see FAMILIES). ``bnd_AP@5`` /
``oth_AP@5`` are the mean AP@5 of the boundary event classes (>= 50 % of the onsets within 2 s of a
phase change, ``EXP2/analysis/oracle_phase_boundary.json``) and of the others, per run.

Experiment 3 (``--exp EXP3``): arms ``outputs/orsi/EXP3/<arm>`` against F_base (``B*``) or FT_gstatic
(``FTB*``) of Experiment 2.

Example:
    python tools/summarize_exp2.py [--exp EXP3] [--wandb]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "outputs/orsi"
CC = "caps_chain|a=0.1|s=1|f=0.2"
FAMILIES = {
    "clip_emostasi": ["clip", "hemostatic", "coagulate"],
    "sutura": ["stitch", "v-lock", "needle", "suture"],
    "oggetti_strumenti": ["port_placement", "instrument_insertion", "bagging", "endobag", "drain", "removal_of_robotic",
                          "grasping_catheter", "visualisation", "leak_test"],
    "incisione_dissezione": ["incision", "dissection", "dissect", "lift", "transection", "adhesion"],
}
LABELS = {
    "2.0_events_only": "2.0 solo eventi (nessuna loss di fase)",
    "2.1_H*": "2.1 eventi + fasi (H*, Esp1)",
    "2.2_phase_cond": "2.2 eventi condizionati alla fase",
    "2.3_oracle_phase": "2.3 fasi vere in input (limite sup.)",
    "S_grasp_phase": "S fasi/step GraSP", "S_grasp_action": "S action/strumenti GraSP", "S_sar": "S gesti SAR",
    "S_esad": "S azioni ESAD", "S_all": "S tutte le fonti",
    "J_grasp_phase": "J fasi/step GraSP", "J_grasp_action": "J action/strumenti GraSP", "J_sar": "J gesti SAR",
    "J_esad": "J azioni ESAD", "J_all": "J tutte le fonti",
    "F_base": "F encoder fine-tunato (base)", "F_S_all": "F fine-tunato + S tutte le fonti",
    "T_base": "T due flussi (base, Esp1)", "T_S_all": "T due flussi + S tutte le fonti",
    "FT_gstatic": "FT clip fine-tunata + frame GraSP", "FT_base": "FT clip + frame fine-tunati",
    "B1_head": "B1 testa di confine (loss)", "B2_add": "B2 + fusione add", "B3_transition": "B3 + fusione transition",
    "B4_oracle_boundary": "B4 confini veri in input (limite sup.)",
}
METRICS = ["mAP@1", "mAP@5", "mAP@30", "tight", "loose", "F1@5", "FA/h", "phase_acc", "cc_mAP@5", "cc_F1@5",
           "bnd_AP@5", "oth_AP@5"]
BOUNDARY = ROOT / "EXP2/analysis/oracle_phase_boundary.json"


def family_of(name):
    """
    >>> family_of("Metal_clip_on_right_pedicle"), family_of("V-lock"), family_of("Start_dissection_DVC")
    ('clip_emostasi', 'sutura', 'incisione_dissezione')
    >>> family_of("Endobag_removal"), family_of("Removing_needles")
    ('oggetti_strumenti', 'sutura')
    """
    n = name.lower()
    for fam, keys in FAMILIES.items():  # order matters: clip/suture before generic dissection words
        if any(k in n for k in keys):
            return fam
    return "altro"


def references(arm, exp="EXP2"):
    """Arms ``arm`` is tested against; the first one is the reference of the table.

    >>> references("S_all"), references("F_S_all"), references("T_S_all"), references("T_base")
    (['2.1_H*'], ['F_base'], ['T_base'], ['2.1_H*'])
    >>> references("FT_base"), references("2.1_H*")
    (['F_base', 'T_base', 'FT_gstatic'], [])
    >>> references("B2_add"), references("FTB2_add"), references("F_base", exp="EXP3")
    (['F_base'], ['FT_gstatic'], [])
    """
    if arm == "2.1_H*" or (exp == "EXP3" and arm in ("F_base", "FT_gstatic")):
        return []
    if arm.startswith("FTB"):
        return ["FT_gstatic"]
    if arm.startswith("B"):
        return ["F_base"]
    if arm.startswith("FT_"):  # fine-tuned clip + a center-frame stream: vs each input alone
        return [r for r in ("F_base", "T_base", "FT_gstatic") if r != arm]
    for prefix in ("F_", "T_"):
        if arm.startswith(prefix) and arm != prefix + "base":
            return [prefix + "base"]
    return ["2.1_H*"]


def reference(arm, exp="EXP2"):
    return (references(arm, exp) or [arm])[0]


def boundary_names():
    return {r["event"] for r in json.loads(BOUNDARY.read_text())["per_class"] if r["boundary_frac"] >= 0.5}


def run_metrics(val, bnd=frozenset()):
    rep = json.loads((val / "report_nms5.json").read_text())
    nms = rep["spotting_nms"]
    op = nms["operating_points"]["delta_5"]["best_f1"]
    m = {f"mAP@{d}": nms["map"][str(d)] for d in (1, 5, 30)}
    m.update(tight=nms["avg_map_tight_1_5s"], loose=nms["avg_map_loose_5_30s"], **{"F1@5": op["f1"]},
             **{"FA/h": op["false_alarms_per_hour"]}, phase_acc=rep["phases_untrimmed"]["accuracy"])
    cpath = val / "constraints_nms5.json"
    if cpath.exists():
        r = json.loads(cpath.read_text())["results"][CC]
        m["cc_mAP@5"], m["cc_F1@5"] = r["map"]["5"], r["best_f1_5s"]["f1"]
    ap = {k: v for k, v in nms["per_class_ap_5s"].items()}
    ok = {k: v for k, v in ap.items() if v is not None and np.isfinite(v)}
    if bnd:
        m["bnd_AP@5"] = float(np.mean([v for k, v in ok.items() if k in bnd]))
        m["oth_AP@5"] = float(np.mean([v for k, v in ok.items() if k not in bnd]))
    return m, ap


def collect(exp="EXP2"):
    if exp == "EXP2":
        arms = {"2.1_H*": ROOT / "HYBRID/G2_dilate3", "T_base": ROOT / "HYBRID/G2_dilate3-two"}
    else:
        arms = {"F_base": ROOT / "EXP2/F_base", "FT_gstatic": ROOT / "EXP2/FT_gstatic"}
    arms.update({p.name: p for p in sorted((ROOT / exp).glob("*")) if p.is_dir() and p.name != "analysis"})
    bnd = boundary_names() if BOUNDARY.exists() else frozenset()
    out = {}
    for arm, root in arms.items():
        runs = {}
        for val in sorted(root.glob("seed*/fold*/val")):
            if not (val / "report_nms5.json").exists():
                continue
            key = (val.parent.parent.name, val.parent.name)
            runs[key] = run_metrics(val, bnd)
        if runs:
            out[arm] = runs
    return out


def summarize(data, exp="EXP2"):
    table, wil, fams = {}, {}, {}
    for arm, runs in data.items():
        ms = [m for m, _ in runs.values()]
        table[arm] = {k: [float(np.mean([m[k] for m in ms if k in m])), float(np.std([m[k] for m in ms if k in m], ddof=1))]
                      for k in METRICS if any(k in m for m in ms)}
        table[arm]["n_runs"] = len(runs)
        per_class = defaultdict(list)
        for _, ap in runs.values():
            for c, v in ap.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    per_class[c].append(v)
        cls = {c: float(np.mean(v)) for c, v in per_class.items()}
        fams[arm] = {f: float(np.mean([v for c, v in cls.items() if family_of(c) == f]))
                     for f in list(FAMILIES) + ["altro"] if any(family_of(c) == f for c in cls)}
        fams[arm]["_per_class"] = cls
    for arm, ref in ((a, r) for a in data for r in references(a, exp)):
        runs = data[arm]
        if ref not in data:
            continue
        common = sorted(set(runs) & set(data[ref]))
        for k in ("mAP@5", "cc_mAP@5", "F1@5", "phase_acc", "bnd_AP@5", "oth_AP@5"):
            a = [runs[r][0].get(k) for r in common]
            b = [data[ref][r][0].get(k) for r in common]
            if len(common) >= 5 and None not in a + b:
                diff = np.array(a) - np.array(b)
                p = wilcoxon(a, b).pvalue if np.any(diff) else 1.0
                wil[f"{arm}|{ref}|{k}"] = {"delta": float(diff.mean()), "p": float(p), "n": len(common),
                                           "wins": int((diff > 0).sum())}
    return table, wil, fams


def markdown(table, wil, fams, exp="EXP2"):
    lines = ["| Braccio | n | mAP@1 | mAP@5 | Δ vs rif. (p) | F1@5 | FA/h | Acc fasi | cc mAP@5 | Δcc (p) "
             "| AP confine | Δ (p) | AP altri | Δ (p) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, t in table.items():
        ref = reference(arm, exp)
        d5 = wil.get(f"{arm}|{ref}|mAP@5")
        dc = wil.get(f"{arm}|{ref}|cc_mAP@5")
        f = lambda k: f"{t[k][0]:.3f}" if k in t else "–"
        lines.append(
            f"| {LABELS.get(arm, arm)} | {t['n_runs']} | {f('mAP@1')} | {t['mAP@5'][0]:.3f} ± {t['mAP@5'][1]:.3f} | "
            + (f"{d5['delta']:+.3f} ({d5['p']:.3f})" if d5 else "–")
            + f" | {f('F1@5')} | {t['FA/h'][0]:.1f} | {f('phase_acc')} | {f('cc_mAP@5')} | "
            + (f"{dc['delta']:+.3f} ({dc['p']:.3f})" if dc else "–")
            + "".join(f" | {f(k)} | " + (f"{w['delta']:+.3f} ({w['p']:.3f})" if (w := wil.get(f"{arm}|{ref}|{k}")) else "–")
                      for k in ("bnd_AP@5", "oth_AP@5")) + " |")
    fam_names = list(FAMILIES) + ["altro"]
    lines += ["", "| Braccio | " + " | ".join(fam_names) + " |", "|---|" + "---|" * len(fam_names)]
    for arm, f in fams.items():
        lines.append(f"| {LABELS.get(arm, arm)} | " + " | ".join(f"{f.get(n, float('nan')):.3f}" for n in fam_names) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="EXP2", choices=["EXP2", "EXP3"])
    ap.add_argument("--out", help="default: outputs/orsi/<exp>")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    data = collect(args.exp)
    table, wil, fams = summarize(data, args.exp)
    out = Path(args.out or ROOT / args.exp)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"table": table, "wilcoxon": wil, "families": fams}, indent=1))
    md = markdown(table, wil, fams, args.exp)
    (out / "summary.md").write_text(md + "\n")
    print(md)
    if args.wandb:
        import wandb

        run = wandb.init(project="OrsiTapis", entity="eidos-giorgio", job_type=f"{args.exp.lower()}_summary",
                         name=f"{args.exp.lower()}/summary")
        for arm, t in table.items():
            run.summary.update({f"{arm}/{k}": v[0] for k, v in t.items() if isinstance(v, list)})
        run.finish()


if __name__ == "__main__":
    main()
