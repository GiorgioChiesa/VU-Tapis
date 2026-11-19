#!/usr/bin/env python3
"""
gpuq: a file-based GPU job queue for ORSI-GPU2 (standard library only).

Jobs are JSON files in ``queue/{pending,running,done,failed}/<id>.json`` (next to this script). Fields:
  id, priority (higher first; ties: earlier enqueue first), deps (list), cmd (bash; ``{gpu}`` = physical GPU),
  cwd, log, gpu (true: needs one GPU; false: CPU job), mem_mb (memory the job needs; a GPU qualifies if
  used + mem_mb <= total), max_used_mb (a GPU qualifies only if less than this is in use, default 6000: tolerates
  another user's idle process), exclusive (true: no compute process of any user on the GPU), group,
  wait_for_group (do not start while any job of that group is pending and not blocked), enqueued.
Dependencies: a job id (must be in done; in failed -> this job is blocked), ``glob:<pattern>`` (something must
match, relative to cwd) or ``count:<n>:<pattern>`` (at least n matches).

Daemon (every 60 s): reap finished jobs (the job's wrapper writes ``<id>.exit``), then start the highest-priority
ready pending job on a free GPU. Free = one of GPUS (1, 2, 3; never 0), no process of user USER, no running
queue job on it, memory below the job's threshold, and seen free at two consecutive ticks. One queue job per GPU.
Jobs run detached (own session): restarting or killing the daemon never stops a job; a restarted daemon picks
up the state from the files. A failed job moves to failed/ and only blocks the jobs that depend on it.
Writes ``results/gpuq.log`` and ``results/STATUS.md`` (worktree root = two levels up).

    python run_files/gpuq/gpuq.py daemon                      # in tmux gpuq
    python run_files/gpuq/gpuq.py add --id myjob --priority 50 --mem 8000 --deps other_job "count:50:results/x/*.npz" \\
        --cmd 'bash run_files/x.sh {gpu}'
    python run_files/gpuq/gpuq.py status
"""

import argparse
import fcntl
import glob
import json
import os
import pwd
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q = ROOT / "queue"
STATES = ("pending", "running", "done", "failed")
WT = ROOT.parent.parent
LOG = WT / "results" / "gpuq.log"
STATUS = WT / "results" / "STATUS.md"
GPUS = (1, 2, 3)
USER = "gchie"
TICK = 60


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[{now()}] {msg}\n")


def jobs(state):
    out = []
    for p in sorted((Q / state).glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    return out


def find(job_id):
    for s in STATES:
        if (Q / s / f"{job_id}.json").exists():
            return s
    return None


def write(job, state):
    p = Q / state / f"{job['id']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1))
    tmp.replace(p)


def move(job, src, dst):
    write(job, dst)
    (Q / src / f"{job['id']}.json").unlink(missing_ok=True)


def dep_state(job):
    """'ready' | 'waiting: <dep>' | 'blocked: <dep>'."""
    for d in job.get("deps", []):
        if d.startswith("glob:"):
            if not glob.glob(d[5:], root_dir=job["cwd"]):
                return f"waiting: {d}"
        elif d.startswith("count:"):
            _, n, pat = d.split(":", 2)
            if len(glob.glob(pat, root_dir=job["cwd"])) < int(n):
                return f"waiting: {d}"
        else:
            s = find(d)
            if s == "failed":
                return f"blocked: {d} failed"
            if s != "done":
                return f"waiting: {d}" + ("" if s else " (unknown job)")
    return "ready"


def gpu_info():
    """{gpu: {"used": MiB, "total": MiB, "users": set of process owners}}."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    info, uuid = {}, {}
    for line in q.strip().splitlines():
        i, u, used, total = [x.strip() for x in line.split(",")]
        info[int(i)] = {"used": int(used), "total": int(total), "users": set()}
        uuid[u] = int(i)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout
    for line in apps.strip().splitlines():
        pid, u = [x.strip() for x in line.split(",")]
        try:
            user = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except (OSError, KeyError):
            user = "?"
        if u in uuid:
            info[uuid[u]]["users"].add(user)
    return info


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie child of this daemon counts as finished
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def reap():
    try:
        while os.waitpid(-1, os.WNOHANG)[0]:
            pass
    except ChildProcessError:
        pass
    for job in jobs("running"):
        ex = Q / "running" / f"{job['id']}.exit"
        if ex.exists():
            try:
                code = int(ex.read_text().strip() or "-1")
            except ValueError:
                code = -1
        elif not alive(job["pid"]):
            time.sleep(2)  # the wrapper may be writing the exit file right now
            if ex.exists():
                continue
            code = None
        else:
            continue
        job.update(end=now(), exit=code)
        dst = "done" if code == 0 else "failed"
        move(job, "running", dst)
        ex.unlink(missing_ok=True)
        log(f"{dst} {job['id']} (gpu {job.get('gpu_id')}, exit {code})")


def start(job, gpu):
    ex = Q / "running" / f"{job['id']}.exit"
    ex.unlink(missing_ok=True)
    cmd = job["cmd"].replace("{gpu}", "" if gpu is None else str(gpu))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), GPUQ_GPU="" if gpu is None else str(gpu),
               GPUQ_JOB=job["id"])
    logp = Path(job["cwd"]) / job["log"]
    logp.parent.mkdir(parents=True, exist_ok=True)
    with open(logp, "a") as lf:
        lf.write(f"[{now()}] gpuq start {job['id']} gpu {gpu}: {cmd}\n")
        lf.flush()
        p = subprocess.Popen(["bash", "-c", 'bash -c "$1"; echo $? > "$2"', "gpuq", cmd, str(ex)], cwd=job["cwd"],
                             env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    job.update(pid=p.pid, gpu_id=gpu, start=now(), cmd_run=cmd)
    move(job, "pending", "running")
    log(f"start {job['id']} (priority {job['priority']}, gpu {gpu}, pid {p.pid})")


def schedule(free_prev):
    """One scheduling pass. Returns the set of GPUs seen free (for the two-tick rule)."""
    try:
        info = gpu_info()
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        log(f"nvidia-smi failed: {e}")
        return set()
    held = {j.get("gpu_id") for j in jobs("running")}
    candidates = {g for g in GPUS if g in info and g not in held and USER not in info[g]["users"]}
    seen_free = candidates & free_prev  # free at this tick and the previous one
    pending = sorted(jobs("pending"), key=lambda j: (-j["priority"], j["enqueued"]))
    states = {j["id"]: dep_state(j) for j in pending}
    active_groups = {j.get("group") for j in pending if not states[j["id"]].startswith("blocked")}
    used_now = set()
    for job in pending:
        if states[job["id"]] != "ready":
            continue
        wg = job.get("wait_for_group")
        if wg and wg in active_groups:
            continue
        if not job.get("gpu", True):
            start(job, None)
            continue
        for g in sorted(seen_free - used_now):
            gi = info[g]
            if gi["used"] >= job.get("max_used_mb", 6000) or gi["used"] + job.get("mem_mb", 0) > gi["total"]:
                continue
            if job.get("exclusive") and gi["users"]:
                continue
            start(job, g)
            used_now.add(g)
            break
    return candidates - used_now


def status_md():
    rows = []
    for s in STATES:
        for j in jobs(s):
            note = dep_state(j) if s == "pending" else ""
            rows.append((s, j))
            j["_note"] = note
    order = {"running": 0, "pending": 1, "failed": 2, "done": 3}
    rows.sort(key=lambda r: (order[r[0]], -r[1]["priority"], r[1]["enqueued"]))
    lines = [f"# GPU queue status ({now()}; daemon tick {TICK} s; log results/gpuq.log)", "",
             "| job | state | prio | gpu | start | end | exit | note | log |", "|---|---|---|---|---|---|---|---|---|"]
    for s, j in rows:
        g = j.get("gpu_id")
        g = "cpu" if not j.get("gpu", True) else ("-" if g is None else g)
        lines.append(f"| {j['id']} | {s} | {j['priority']} | {g} | {j.get('start', '')} | {j.get('end', '')} | "
                     f"{'' if j.get('exit') is None else j['exit']} | {j['_note']} | {j['log']} |")
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(STATUS)
    return "\n".join(lines)


def daemon():
    for s in STATES:
        (Q / s).mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / "daemon.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another gpuq daemon is running")
    log(f"daemon start (pid {os.getpid()}, gpus {GPUS}, tick {TICK} s)")
    free_prev = set()
    while True:
        try:
            reap()
            free_prev = schedule(free_prev)
            status_md()
        except Exception as e:  # keep the daemon alive; the error is logged
            log(f"daemon error: {type(e).__name__}: {e}")
        time.sleep(TICK)


def add(a):
    for s in STATES:
        (Q / s).mkdir(parents=True, exist_ok=True)
    if find(a.id):
        sys.exit(f"job {a.id} already exists ({find(a.id)}); remove its file to re-enqueue")
    cwd = str(Path(a.cwd).resolve())
    job = {"id": a.id, "priority": a.priority, "deps": a.deps, "cmd": a.cmd, "cwd": cwd,
           "log": a.log or f"results/gpuq_logs/{a.id}.log", "gpu": not a.cpu, "mem_mb": a.mem,
           "max_used_mb": a.max_used, "exclusive": a.exclusive, "group": a.group,
           "wait_for_group": a.wait_for_group, "enqueued": now()}
    write(job, "pending")
    log(f"enqueued {a.id} (priority {a.priority}, deps {a.deps})")
    print(f"enqueued {a.id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd_", required=True)
    sub.add_parser("daemon")
    sub.add_parser("status")
    a = sub.add_parser("add")
    a.add_argument("--id", required=True)
    a.add_argument("--priority", type=int, default=50)
    a.add_argument("--deps", nargs="*", default=[])
    a.add_argument("--cmd", required=True, help="bash command; {gpu} = physical GPU id")
    a.add_argument("--cwd", default=str(WT))
    a.add_argument("--log", help="relative to cwd (default results/gpuq_logs/<id>.log)")
    a.add_argument("--cpu", action="store_true", help="no GPU needed")
    a.add_argument("--mem", type=int, default=8000, help="GPU memory needed (MiB)")
    a.add_argument("--max-used", type=int, default=6000, help="GPU qualifies only below this memory in use (MiB)")
    a.add_argument("--exclusive", action="store_true", help="no compute process of any user on the GPU")
    a.add_argument("--group", default="")
    a.add_argument("--wait-for-group", default="")
    args = ap.parse_args()
    if args.cmd_ == "daemon":
        daemon()
    elif args.cmd_ == "status":
        print(status_md())
    else:
        add(args)


if __name__ == "__main__":
    main()
