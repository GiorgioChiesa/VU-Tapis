#!/usr/bin/env python3
"""
Backfill W&B runs with the training loss of finished encoder fine-tunes (cv_finetune.py folds).

The live fine-tune runs (`<name>_fold<i>`, logged by tools/train_net.py) carry the validation
metrics but no training loss. The loss is only on disk, as one `train_iter` json_stats record per
epoch in `<fold>/stdout.log` (median of the epoch's iteration losses). For every finished fold
(`.finetune.done`) this script creates one run `<prefix>_<arm>_fold<i>_finetune` (group, tags
arm/fold) with, per epoch (W&B step = epoch):
  train/loss, train/loss_<task>, train/lr            from stdout.log
  train/<task>_mAP                                   from the live run
  val/<task>_mAP, val/<task>_f1_score, val/<task>_balanced_accuracy   from the live run (epoch 0 =
                                                     evaluation before training)
Folds that are still training, or whose backfill run already exists, are skipped (listed at the end).
W&B login comes from ~/.netrc.

  python tools/wandb_backfill_finetune.py --arms L A0aug S          # all finished folds
  python tools/wandb_backfill_finetune.py --arms A0aug --folds 3 4 --dry-run
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JSON_RE = re.compile(r"json_stats: (\{.*\})\s*$")
VAL_KEYS = ("mAP", "f1_score", "balanced_accuracy")


def train_loss(stdout_log):
    """{epoch (1-based): {loss, loss_<task>, lr}}; a resumed run keeps the last record per epoch."""
    out = {}
    for line in Path(stdout_log).read_text(errors="replace").splitlines():
        m = JSON_RE.search(line)
        if not m:
            continue
        rec = json.loads(m.group(1))
        if rec.get("_type") != "train_iter":
            continue
        row = {"train/loss": float(rec["overall_loss"]), "train/lr": float(rec["lr"])}
        row.update({f"train/{k}": float(v) for k, v in rec.items() if k.startswith("loss_")})
        out[int(rec["cur_epoch"])] = row
    return out


def live_run(api, path, name, fold_dir):
    """The live fine-tune run of this fold (same name, history pointing at this output dir)."""
    best = None
    for r in api.runs(path, filters={"display_name": name}):
        for row in r.scan_history(keys=["train_output_dir"], page_size=1000):
            if row.get("train_output_dir") == str(fold_dir):
                if best is None or (r.lastHistoryStep or 0) > (best.lastHistoryStep or 0):
                    best = r
                break
    return best


def live_metrics(run, tasks):
    """{epoch: {train/<task>_mAP, val/<task>_<metric>}} from the live run history."""
    out = defaultdict(dict)
    for row in run.scan_history(page_size=1000):
        if "cur_epoch" not in row or row["cur_epoch"] is None:
            continue
        e = int(row["cur_epoch"])
        for t in tasks:
            for k in VAL_KEYS:
                v = row.get(f"val/{t}_{k}")
                if v is not None:
                    out[e][f"val/{t}_{k}"] = float(v)
            v = row.get(f"train_{t}_map.{t}_mAP")
            if v is not None:
                out[e][f"train/{t}_mAP"] = float(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="<results>/<arm>/LONG/cv_ft/fold<i>")
    p.add_argument("--arms", nargs="+", default=["L", "A0aug", "S"])
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--prefix", default="exp19", help="live runs are <prefix>_<arm>_fold<i>")
    p.add_argument("--group", default="EXP-19")
    p.add_argument("--entity", default="eidos-giorgio")
    p.add_argument("--project", default="OrsiTapis")
    p.add_argument("--tasks", nargs="+", default=["steps", "phases"])
    p.add_argument("--dry-run", action="store_true", help="parse and print, log nothing")
    args = p.parse_args()

    import wandb

    api = wandb.Api(timeout=60)
    path = f"{args.entity}/{args.project}"
    done, skipped = [], []
    for arm in args.arms:
        for i in args.folds:
            fold_dir = (REPO / args.results / arm / "LONG" / "cv_ft" / f"fold{i}").resolve()
            name = f"{args.prefix}_{arm}_fold{i}_finetune"
            if not fold_dir.exists():
                continue
            if not (fold_dir / ".finetune.done").exists():
                skipped.append(f"{name} (fine-tune not finished)")
                continue
            if any(True for _ in api.runs(path, filters={"display_name": name})):
                skipped.append(f"{name} (already in W&B)")
                continue
            loss = train_loss(fold_dir / "stdout.log")
            live = live_run(api, path, f"{args.prefix}_{arm}_fold{i}", fold_dir)
            metrics = live_metrics(live, args.tasks) if live else {}
            epochs = sorted(set(loss) | set(metrics))
            print(f"{name}: {len(loss)} train-loss epochs, {len(metrics)} metric epochs,"
                  f" live run {live.id if live else 'not found'}", flush=True)
            if args.dry_run:
                continue
            run = wandb.init(
                entity=args.entity, project=args.project, name=name, group=args.group,
                tags=[f"arm:{arm}", f"fold:{i}", "finetune", "backfill"], job_type="finetune_backfill",
                config={"arm": arm, "fold": i, "output_dir": str(fold_dir),
                        "live_run": live.id if live else None,
                        "source": "stdout.log train_iter json_stats (epoch median loss), live run history (metrics)"},
            )
            run.define_metric("epoch")
            run.define_metric("*", step_metric="epoch")
            for e in epochs:
                row = {"epoch": e, **loss.get(e, {}), **metrics.get(e, {})}
                run.log(row, step=e)
            for t in args.tasks:
                val = {e: m[f"val/{t}_mAP"] for e, m in metrics.items() if f"val/{t}_mAP" in m}
                if val:
                    best = max(val, key=val.get)
                    run.summary[f"best_val/{t}_mAP"] = val[best]
                    run.summary[f"best_val/{t}_epoch"] = best
            run.finish()
            done.append(name)
    print("logged:", " ".join(done) or "none")
    print("skipped:", "; ".join(skipped) or "none")


if __name__ == "__main__":
    main()
