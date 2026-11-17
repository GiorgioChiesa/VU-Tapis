#!/usr/bin/env python3
"""
Video-level cross-validation with the MViT encoder fine-tuned per fold.

For each fold (same folds as `train_temporal.py --cv-fold`, i.e. patients sorted
and shuffled with Random(1234), fold i = every 5th starting at i):

  1. fine-tune TAPIS/MViT with the configuration of a reference run (default:
     `control`) on the 40 non-held-out videos: 35 train + 5 inner validation used
     only to pick `checkpoint_best_steps.pyth`; the held-out videos are never seen;
  2. extract per-second features of all 50 videos with that checkpoint;
  3. train the temporal heads (one per target) with `--cv-fold i/5`;
  4. run eval_untrimmed / eval_constraints on the held-out videos, both for the
     temporal heads and for the per-clip MViT itself.

Folds are distributed over the given GPUs (one fold at a time per GPU). Every step
writes a `.done` marker, so the script can be relaunched and resumes.

Variants (defaults reproduce the historical run): ``--label-dir`` trains and evaluates on other
label CSVs (DATA.LABEL_DIR); ``--fps 4 --fps-frame-dir DIR`` fine-tunes and extracts features on the
4 fps frame cache (DATA.FPS / DATA.FRAME_DIR, ``--label-dir`` must then be the grid labels);
``--set KEY VALUE ...`` overrides config keys (e.g. ``ENDOVIS_DATASET.ADD_IDLE 20``);
``--targets``, ``--seeds``, ``--temporal-args`` and ``--eval-args`` shape the temporal heads and
their evaluation (seed s > 0 writes ``cv_ft_<target>_s<s>``).

Example:
    nohup setsid python tools/cv_finetune.py --gpus 0 1 > outputs/untrimmed_logs/cv_finetune.log 2>&1 &
"""

import argparse
import os
import queue
import random
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
TARGETS = {"dilate3": ["--target", "dilate", "--k", "3"], "asym3": ["--target", "asym", "--k", "3"]}


def folds(patients, n=5, seed=1234):
    shuffled = sorted(patients)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::n] for i in range(n)]


def class_ids(cfg, split, train, val):
    """Class order `Orsi` rebuilds for `split` (it depends on which events occur in it)."""
    import contextlib
    import io

    from tapis.datasets.orsi import Orsi

    cfg.ENDOVIS_DATASET.TRAIN_LISTS = [f"{p}.csv" for p in train]
    cfg.ENDOVIS_DATASET.VAL_LISTS = [f"{p}.csv" for p in val]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = Orsi(cfg, split, load=False)
            ds._load_data()
            ds.generate_weight_vector()
    except AssertionError:
        return None
    return {task: list(ds.counter[task]["id"]) for task in ds.counter}


def choose_inner_val(rest, i, n, check_cfg):
    """
    First deterministic sample of `n` videos whose rebuilt class order equals the
    training one: `Orsi.collapse_event_dfs` only collapses events whose target is
    present, so a small validation split can end up with a different class list.
    """
    for attempt in range(200):
        val = sorted(random.Random(100 + i + 1000 * attempt).sample(rest, n))
        train = [p for p in rest if p not in val]
        ids_train = class_ids(check_cfg, "train", train, val)
        if ids_train is not None and class_ids(check_cfg, "val", train, val) == ids_train:
            return val, train
    raise RuntimeError(f"fold {i}: no inner validation split with consistent classes")


def host_paths(ens, data_dir):
    for key in ("FRAME_DIR", "ORSI_ROOT_DIR", "ORSI_LABEL_DIR"):
        ens[key] = f"{data_dir}/orsi_tensors"
    for key in ("FRAME_LIST_DIR", "ANNOTATION_DIR"):
        ens[key] = f"{data_dir}/coco"
    ens["VAL_COCO_ANNS"] = f"{data_dir}/coco/all_merged.json"


def apply_overrides(cfg, pairs):
    """
    ``KEY.SUB VALUE`` pairs onto the nested config dict (values parsed as YAML).

    >>> apply_overrides({"A": {"B": 1}}, ["A.B", "20", "A.C", "[1, 2]"])
    {'A': {'B': 20, 'C': [1, 2]}}
    """
    if len(pairs) % 2:
        raise ValueError(f"--set needs KEY VALUE pairs: {pairs}")
    for key, val in zip(pairs[::2], pairs[1::2]):
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def wait_gpu_free(gpu, max_used_mib=1500, poll=60):
    """Block until the physical GPU has (almost) no memory in use, e.g. a previous job ended."""
    import time

    announced = False
    while True:
        used = int(
            subprocess.run(
                ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip() or 0
        )
        if used < max_used_mib:
            return
        if not announced:
            print(f"[gpu{gpu}] waiting: {used} MiB in use", flush=True)
            announced = True
        time.sleep(poll)


def run(cmd, log, gpu, done):
    if done.exists():
        print(f"[gpu{gpu}] skip (done): {done.parent.name}/{done.name}", flush=True)
        return
    wait_gpu_free(gpu)
    print(f"[gpu{gpu}] {' '.join(map(str, cmd))[:200]}", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_SILENT="true")
    with open(log, "w") as f:
        rc = subprocess.run(list(map(str, cmd)), cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"failed ({rc}): see {log}")
    done.touch()


def fold_job(i, args, base_cfg, all_folds, gpu, splits):
    held_out = all_folds[i]
    inner_val, train = splits[i]

    root = REPO / args.out / f"fold{i}"
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    cfg = yaml.safe_load(base_cfg)
    ens = cfg["ENDOVIS_DATASET"]
    host_paths(ens, args.data_dir)
    ens["TRAIN_LISTS"] = [f"{p}.csv" for p in train]
    ens["VAL_LISTS"] = [f"{p}.csv" for p in inner_val]
    ens["TEST_LISTS"] = [f"{p}.csv" for p in held_out]
    ens["TRAIN_GT_BOX_JSON"] = [f"{p}_coco.json" for p in train]
    ens["VAL_GT_BOX_JSON"] = [f"{p}_coco.json" for p in inner_val]
    ens["TEST_GT_BOX_JSON"] = [f"{p}_coco.json" for p in held_out]
    cfg["OUTPUT_DIR"] = str(root)
    cfg["NAME"] = f"{args.name}_fold{i}"
    # assert_and_infer_cfg() overwrites CUDA_VISIBLE_DEVICES with GPUIDS: use the physical id
    cfg["GPUIDS"] = [gpu]
    cfg["TEST"]["ENABLE"] = False
    cfg["WANDB_ENABLE"] = args.wandb
    cfg["WANDB_API"] = ""  # login comes from ~/.netrc; never store the key in configs/logs
    cfg["TRAIN"]["CHECKPOINT_FILE_PATH"] = args.init_ckpt
    if args.label_dir:
        cfg["DATA"]["LABEL_DIR"] = args.label_dir
    if args.fps != 1:
        cfg["DATA"]["FPS"] = args.fps
        cfg["DATA"]["FRAME_DIR"] = args.fps_frame_dir
    apply_overrides(cfg, args.set)
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # 1. fine-tune the encoder (a run that reached the last epoch outside this script counts as done)
    last = root / "checkpoints" / f"checkpoint_epoch_{cfg['SOLVER']['MAX_EPOCH']:05d}.pyth"
    if last.exists() and (root / "checkpoint_best_steps.pyth").exists():
        (root / ".finetune.done").touch()
    run([PY, "-B", "tools/run_net_host.py", "--cfg", cfg_path, "WANDB_API", '""'],
        logs / "finetune.log", gpu, root / ".finetune.done")
    # 2. features of all videos
    feat_dir = root / "untrimmed" / "checkpoint_best_steps" / "all"
    fps_args = (["--fps", args.fps, "--label-dir", args.label_dir, "--frame-dir", args.fps_frame_dir]
                if args.fps != 1 else ["--frame-dir", f"{args.data_dir}/orsi_tensors"])
    run([PY, "-W", "ignore", "tools/infer_untrimmed.py", "--run", root, "--ckpt", "checkpoint_best_steps.pyth",
         "--split", "all", "--features", *fps_args, "--gpu", 0],
        logs / "features.log", gpu, root / ".features.done")
    # per-clip MViT on the held-out videos
    evals = [(feat_dir, held_out, "mvit")]
    # 3. temporal heads
    for target in args.targets:
        for seed in args.seeds:
            name = target + (f"_s{seed}" if seed else "")
            seed_args = ["--seed", seed] if seed else []
            out = REPO / args.temporal_out / f"cv_ft_{name}" / f"fold{i}"
            run([PY, "-W", "ignore", "tools/train_temporal.py", "--feat-dir", feat_dir, *TARGETS[target], "--cv-fold",
                 f"{i}/5", "--out", out, "--gpu", 0, *seed_args, *shlex.split(args.temporal_args),
                 *(["--wandb"] if args.wandb else [])],
                logs / f"temporal_{name}.log", gpu, root / f".temporal_{name}.done")
            evals.append((out / "val", None, name))
    # 4. evaluation
    for pred_dir, patients, name in evals:
        pts = ["--patients", *patients] if patients else []
        ev_args = shlex.split(args.eval_args)
        run([PY, "-W", "ignore", "tools/eval_untrimmed.py", "--pred-dir", pred_dir,
             "--frame-dir", f"{args.data_dir}/orsi_tensors", *pts, *ev_args],
            logs / f"eval_{name}.log", gpu, root / f".eval_{name}.done")
        if name != "mvit":
            run([PY, "-W", "ignore", "tools/eval_constraints.py", "--pred-dir", pred_dir,
                 "--frame-dir", f"{args.data_dir}/orsi_tensors", "--alpha", "0.1", "--phase-smooth", "1",
                 "--gate-floor", "0.2", *ev_args],
                logs / f"constraints_{name}.log", gpu, root / f".constraints_{name}.done")
    print(f"[gpu{gpu}] fold {i} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ref-run", default="outputs/orsi/LONG/control/totale")
    parser.add_argument("--ref-ckpt", default="checkpoint_best_steps.pyth")
    parser.add_argument("--init-ckpt", default="data/pretrained_models/train/LONG.pyth")
    parser.add_argument("--data-dir", default="/home/gchie/workspace/data")
    parser.add_argument("--out", default="outputs/orsi/LONG/cv_ft")
    parser.add_argument("--temporal-out", default="outputs/orsi/TEMPORAL")
    parser.add_argument("--inner-val", type=int, default=5)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--name", default="cv_ft", help="run name prefix (cfg NAME = <name>_fold<i>)")
    parser.add_argument("--label-dir", default="", help="DATA.LABEL_DIR: <dir>/RARPxx/RARPxx_all_labels.csv")
    parser.add_argument("--fps", type=int, default=1, help="DATA.FPS (>1 needs --fps-frame-dir and grid labels)")
    parser.add_argument("--fps-frame-dir", help="DATA.FRAME_DIR: <dir>/RARPxx/frames_<fps>fps.npy")
    parser.add_argument("--set", nargs="*", default=[], help="KEY VALUE config overrides, e.g. ENDOVIS_DATASET.ADD_IDLE 20")
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="temporal-head seeds")
    parser.add_argument("--temporal-args", default="", help="extra train_temporal.py arguments (one string)")
    parser.add_argument("--eval-args", default="", help="extra eval_untrimmed/eval_constraints arguments (one string)")
    args = parser.parse_args()
    if args.fps != 1 and not (args.fps_frame_dir and args.label_dir):
        parser.error("--fps > 1 needs --fps-frame-dir and --label-dir")

    base_cfg = torch.load(REPO / args.ref_run / args.ref_ckpt, map_location="cpu", weights_only=False)["cfg"]
    ens = yaml.safe_load(base_cfg)["ENDOVIS_DATASET"]
    patients = [Path(f).name.removesuffix(".csv") for k in ("TRAIN_LISTS", "VAL_LISTS", "TEST_LISTS") for f in ens[k]]
    all_folds = folds(patients)

    sys.path.insert(0, str(REPO / "tools"))
    from eval_untrimmed import load_cfg

    check_cfg = load_cfg(REPO / args.ref_run, args.ref_ckpt)
    check_cfg.defrost()
    check_cfg.ENDOVIS_DATASET.FRAME_DIR = check_cfg.ENDOVIS_DATASET.ORSI_ROOT_DIR = f"{args.data_dir}/orsi_tensors"
    check_cfg.DATA.LABEL_DIR = args.label_dir
    check_cfg.OUTPUT_DIR = str(REPO / args.out)
    (REPO / args.out).mkdir(parents=True, exist_ok=True)
    splits = {}
    for i in args.folds:
        rest = sorted(p for j, f in enumerate(all_folds) if j != i for p in f)
        splits[i] = choose_inner_val(rest, i, args.inner_val, check_cfg)
        print(f"fold {i}: held-out {all_folds[i]} inner-val {splits[i][0]}", flush=True)

    jobs = queue.Queue()
    for i in args.folds:
        jobs.put(i)
    errors = []

    def worker(gpu):
        while True:
            try:
                i = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                fold_job(i, args, base_cfg, all_folds, gpu, splits)
            except Exception as e:  # keep the other folds going
                errors.append((i, str(e)))
                print(f"[gpu{gpu}] fold {i} FAILED: {e}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("errors:", errors if errors else "none", flush=True)


if __name__ == "__main__":
    main()
