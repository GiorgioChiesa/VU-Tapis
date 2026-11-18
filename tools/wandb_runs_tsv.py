#!/usr/bin/env python3
"""
Rebuild the table of the W&B runs of one experiment, one row per run: time, arm, fold, step, seed, url.

  fine-tune        <prefix>_<arm>_fold<i>             (tools/train_net.py, live)
  finetune_backfill <prefix>_<arm>_fold<i>_finetune   (tools/wandb_backfill_finetune.py)
  temporal_<target> train_temporal.py runs whose --out is <results>/<arm>/TEMPORAL/cv_ft_<target>[_s<seed>]/fold<i>

The file is rewritten from the W&B API every time (idempotent), so it also lists the runs of jobs
started before this script existed. W&B login comes from ~/.netrc.

  python tools/wandb_runs_tsv.py            # -> results/wandb_runs.tsv
"""

import argparse
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--prefix", default="exp19")
    p.add_argument("--out", default="results/wandb_runs.tsv")
    p.add_argument("--entity", default="eidos-giorgio")
    p.add_argument("--project", default="OrsiTapis")
    args = p.parse_args()

    import wandb

    api = wandb.Api(timeout=60)
    path = f"{args.entity}/{args.project}"
    results = (REPO / args.results).resolve()
    ft_re = re.compile(rf"^{re.escape(args.prefix)}_(.+)_fold(\d+)(_finetune)?$")
    tp_re = re.compile(rf"^{re.escape(str(results))}/([^/]+)/TEMPORAL/cv_ft_(.+?)(?:_s(\d+))?/fold(\d+)$")

    rows = []
    for r in api.runs(path, filters={"display_name": {"$regex": f"^{re.escape(args.prefix)}_"}}):
        m = ft_re.match(r.name)
        if m:
            step = "finetune_backfill" if m.group(3) else "finetune"
            rows.append((r.created_at, m.group(1), m.group(2), step, "-", r.url))
    for r in api.runs(path, filters={"config.out": {"$regex": f"^{re.escape(str(results))}/"}}):
        out = api.run(f"{path}/{r.id}").config.get("out", "")  # listings do not always carry the config
        m = tp_re.match(str(out))
        if m:
            arm, target, seed, fold = m.groups()
            rows.append((r.created_at, arm, fold, f"temporal_{target}", seed or "0", r.url))

    rows.sort()
    out = REPO / args.out
    tmp = out.with_suffix(f".{os.getpid()}.tmp")  # lanes may rebuild it concurrently
    tmp.write_text("time\tarm\tfold\tstep\tseed\turl\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    os.replace(tmp, out)
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
