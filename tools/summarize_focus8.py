#!/usr/bin/env python3
"""
Markdown tables from tools/eval_focus8.py reports (aggregates only).

    python tools/summarize_focus8.py --variant nms LABEL=path/eval_test.json ...
Group table: short / long mAP@1/2/5 with the per-video bootstrap 95% CI; per-class table: AP@1/2/5;
phase table: row accuracy, macro F1, F1@50 (mean per video) P0 / P1.
"""

import argparse
import json
from pathlib import Path

SHORT_ABBR = {"Grasping_catheter_tip": "Grasp", "Clip_or_coagulate_vas_deferens_-_right": "ClipVas",
              "Identification_and_clipping_of_SV_arteries_-_right": "SVart", "Metal_clip_on_right_pedicle": "MetalClip",
              "Start_dissection": "StartDiss", "Incision_of_Denonvilliers_fascia": "Denonv",
              "Transection_of_the_urethra": "Transect", "Stitch_in_bladder": "StitchBl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=["nms"])
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("reports", nargs="+", help="LABEL=path")
    args = ap.parse_args()
    rows = [(s.split("=", 1)[0], json.loads(Path(s.split("=", 1)[1]).read_text())) for s in args.reports]
    print("| predictions | decoder | short mAP@1 / @2 / @5 [95% CI] | long mAP@1 / @2 / @5 [95% CI] |")
    print("|---|---|---|---|")
    for lab, r in rows:
        for v in args.variant:
            if v not in r:
                continue
            gm, ci = r[v]["group_map"], r[v].get("group_map_ci95", {})
            cells = []
            for g in ("short", "long"):
                if g not in gm:
                    cells.append("-")
                    continue
                cells.append(" / ".join(f"{gm[g][d]:.3f}" + (f" [{ci[g][d][0]:.2f}, {ci[g][d][1]:.2f}]" if ci else "")
                                        for d in ("1", "2", "5")))
            print(f"| {lab} | {v} | {cells[0]} | {cells[1]} |")
    if args.per_class:
        print()
        names = [n for n in SHORT_ABBR]
        print("| predictions | decoder | " + " | ".join(SHORT_ABBR[n] for n in names) + " |  (AP@1/@2/@5)")
        print("|---|---|" + "---|" * len(names))
        for lab, r in rows:
            for v in args.variant:
                if v not in r:
                    continue
                pc = r[v]["per_class_ap"]
                cells = [" ".join(f"{pc[d][n]:.2f}" for d in ("1", "2", "5")) if n in pc["1"] else "-" for n in names]
                print(f"| {lab} | {v} | " + " | ".join(cells) + " |")
    print()
    print("| predictions | row acc | row macro F1 | F1@50 P0 | F1@50 P1 |")
    print("|---|---|---|---|---|")
    for lab, r in rows:
        if "phases" in r:
            p = r["phases"]
            print(f"| {lab} | {p['row_accuracy']:.3f} | {p['row_macro_f1']:.3f} | {p['F1@50_mean_video']['P0']:.1f} | "
                  f"{p['F1@50_mean_video']['P1']:.1f} |")


if __name__ == "__main__":
    main()
