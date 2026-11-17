#!/usr/bin/env python3
"""
Revised label CSVs from the reviews of the label-review web app.

The app (``tools/label_review.html``, served by ``tools/serve_viewer.py --api``) appends one JSON
line per decision to ``revisions.jsonl`` (append only, the history stays). For each item the last
non-skipped line wins; a skip never undoes an earlier decision.

Output: ``<out>/RARPxx/RARPxx_all_labels.csv`` for every patient, same columns as the original
``RARPxx_all_labels.csv`` (never modified) plus

  * ``revision``:    none | class | moved | both | moved_from
  * ``orig_second``: second the event of the row had in the original CSV

A class change rewrites ``event_id``/``event_name``/``classes_id``/``classes_name`` of the row.
A move clears the original row (it becomes Idle, ``revision=moved_from``) and writes the event on
the row of second ``round(new_time_s)``: the Idle row of that second, or a new row copied from it
when the second already holds an event (the original CSVs also have two events in one second).
The exact sub-second time stays in ``revisions.jsonl``.

Optional extensions (off by default, the default output is unchanged):

  * ``--skipped ignore``: an item whose effective decision is a skip loses its event (the row
    becomes Idle, ``revision=skipped``) and every row within ``--ignore-sec`` of its time gets
    ``ignore=1`` (to be left out of losses and metrics); other rows get ``ignore=0``;
  * ``--times``: column ``time_s``, the fractional time of every event row: ``new_time_s`` of the
    review for reviewed items, else the raw stamp of ``annotations/RARPxx/SURGICAL_PHASES/RARPxx.json``
    (``event_times.stamp_for``, same raw name and ``round(t) == second``), else the second; Idle rows
    get their second;
  * ``--out-4hz DIR``: the same labels on a 0.25 s grid (rows ``k = 0 .. 4 * n_sec - 1``, matching
    ``tools/build_4fps_frames.py``): Idle rows carry the phase of second ``k // 4``, each event sits on
    row ``round(time_s * 4)`` (an extra row when that row already holds one), ``frame_id = k``,
    ``second = k / 4``, ``ignore`` recomputed on the grid. Needs ``--times``;
  * ``--report``: aggregate counts (per action, per class, shift distribution), no raw times.

Example (Vu-Tapis env):
    python tools/build_revised_labels.py
    python tools/build_revised_labels.py --skipped ignore --ignore-sec 3 --times \
        --out data/orsi_labels_revised/v2026-11-17 --out-4hz data/orsi_labels_revised/v2026-11-17_4hz \
        --report data/orsi_labels_revised/v2026-11-17/report.json
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ANN_DIR = Path("/home/gchie/workspace/nas_private/data/orsi/annotations")
REVIEW_DIR = Path("/home/gchie/workspace/viewer/revisione_etichette")
LABEL_DIR = Path("/home/gchie/workspace/data/orsi_tensors")
OUT_DIR = Path("/home/gchie/workspace/data/orsi_labels_revised")
FPS = 4
EVENT_COLS = ["event_id", "event_name", "classes_id", "classes_name"]
ACTIONS = ("confirmed", "changed", "moved", "changed+moved", "skipped")


# ------------------------------------------------------------------------------- jsonl store
def action_of(original_class, new_class, offset, skipped=False):
    """
    >>> action_of(3, 3, 0), action_of(3, 4, 0), action_of(3, 3, -2), action_of(3, 4, 1)
    ('confirmed', 'changed', 'moved', 'changed+moved')
    """
    if skipped:
        return "skipped"
    changed, moved = new_class != original_class, offset != 0
    return {(False, False): "confirmed", (True, False): "changed",
            (False, True): "moved", (True, True): "changed+moved"}[changed, moved]


def make_record(item, new_class, offset, skipped, n_classes, max_offset=(-32, 31), timestamp=None):
    """The jsonl line of one decision on ``item`` (an entry of ``items.json``)."""
    new_class, offset = int(new_class), int(offset)
    if not 0 <= new_class < n_classes:
        raise ValueError(f"class {new_class} out of range")
    if not max_offset[0] <= offset <= max_offset[1]:
        raise ValueError(f"offset {offset} outside the 64-frame window")
    if skipped:
        new_class, offset = int(item["true"]), 0
    return dict(
        id=item["id"], patient=item["patient"], original_second=int(item["second"]),
        original_class=int(item["true"]), new_class=new_class, offset_frames_4fps=offset,
        new_time_s=max(0.0, item["second"] + offset / FPS),
        action=action_of(int(item["true"]), new_class, offset, skipped),
        timestamp=timestamp,
    )


def append_revision(path, record):
    """Append one line under an exclusive lock, flushed and fsynced before the lock is released."""
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_revisions(path):
    """All lines in order; a torn last line (crash mid-write) is ignored."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        lines = f.read().splitlines()
        fcntl.flock(f, fcntl.LOCK_UN)
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return out


def effective_revisions(records):
    """id -> record: the last non-skipped one, else the last skip."""
    eff = {}
    for r in records:
        if r["action"] != "skipped" or eff.get(r["id"], {}).get("action", "skipped") == "skipped":
            eff[r["id"]] = r
    return eff


# ---------------------------------------------------------------------------------- builder
def round_half_up(x):
    return int(math.floor(x + 0.5))


def class_values(meta, events_lookup):
    """class index -> {event_id, event_name, classes_id, classes_name} for the CSV rows."""
    out = {}
    for c, (gid, name) in enumerate(zip(meta["class_global_ids"], meta["class_csv_names"])):
        if gid not in events_lookup:
            raise KeyError(f"class {c} ({name}): global id {gid} not in the label CSVs")
        out[c] = dict(event_id=gid, event_name=name, **events_lookup[gid])
    return out


def events_lookup_of(dfs):
    """global event_id -> its classes_id / classes_name (constant per event in the CSVs)."""
    lk = {}
    for df in dfs:
        for gid, cid, cname in df[["event_id", "classes_id", "classes_name"]].drop_duplicates().itertuples(index=False):
            lk.setdefault(int(gid), dict(classes_id=int(cid), classes_name=cname))
    return lk


# explicit merges of ``Orsi.collapse_event_dfs`` (tapis/datasets/orsi.py); left/right are merged too
COLLAPSE = {
    "instrument_swap:_removal": "removal_of_robotic_instruments",
    "hemolock_clip_on_right_pedicle": "metal_clip_on_right_pedicle",
    "hemolock_clip_on_left_pedicle": "metal_clip_on_left_pedicle",
    "insert_gauze": "insert_hemostatic_agens",
    "cutting_the_needles": "removing_needles",
}


def collapse_key(name):
    """
    Name key under which ``collapse_event_dfs`` merges events.

    >>> collapse_key("Incision peritoneum - left") == collapse_key("Incision_peritoneum_-_right")
    True
    >>> collapse_key("Instrument swap: removal") == collapse_key("Removal of robotic instruments")
    True
    """
    n = str(name).strip().lower().replace(" ", "_")
    n = COLLAPSE.get(n, n)
    return n.replace("left", "").replace("right", "")


def find_source_row(df, rec, meta):
    """
    Index of the CSV row an item comes from: among the rows of its second (Idle or not as the
    item), the one with the class's global id, else the one whose name collapses to the class,
    else the only one.
    """
    at = df.index[df["second"] == rec["original_second"]]
    idle_gid = meta["class_global_ids"][meta["idle"]]
    is_event = rec["original_class"] != meta["idle"]
    cand = [i for i in at if (df.at[i, "event_id"] != idle_gid) == is_event]
    for match in (lambda i: df.at[i, "event_id"] == meta["class_global_ids"][rec["original_class"]],
                  lambda i: collapse_key(df.at[i, "event_name"]) == collapse_key(meta["class_csv_names"][rec["original_class"]]),
                  lambda i: True):
        hit = [i for i in cand if match(i)]
        if len(hit) == 1:
            return hit[0]
    raise LookupError(f"{rec['id']}: {len(cand)} candidate rows at second {rec['original_second']}")


def revise_patient(df, recs, meta, values, skipped="keep", times=False, confirmed_time="review"):
    """
    Revised copy of one patient's CSV (``df`` untouched) and a dict of counts.
    ``recs``: effective revisions of this patient; ``values``: ``class_values``.
    Sources are resolved and cleared first, then moved events are written, so an event moved
    onto the second another event just left lands on that (now Idle) row.
    ``skipped="ignore"``: skipped items become Idle (``revision=skipped``, raw name kept in
    ``orig_event_name``); ``times``: column ``time_s`` = ``new_time_s`` on reviewed event rows
    (NaN elsewhere, filled by `add_times_and_ignore`); ``confirmed_time="raw"`` leaves confirmed
    rows to the raw stamp instead (the review grid puts them on the integer second).
    """
    df = df.copy()
    df["revision"] = "none"
    df["orig_second"] = df["second"].astype(int)
    if times:
        df["time_s"] = np.nan
    if skipped == "ignore":
        df["orig_event_name"] = ""
    idle = meta["idle"]
    lo, hi = int(df["second"].min()), int(df["second"].max())
    n = dict(confirmed=0, class_changed=0, moved=0, moved_same_second=0, rows_added=0)
    srcs = [(rec, find_source_row(df, rec, meta)) for rec in recs if rec["action"] != "skipped"]
    if skipped == "ignore":
        n["skipped_removed"] = 0
        for rec in recs:
            if rec["action"] == "skipped":
                i = find_source_row(df, rec, meta)
                df.at[i, "orig_event_name"] = df.at[i, "event_name"]
                for k, v in values[idle].items():
                    df.at[i, k] = v
                df.at[i, "revision"] = "skipped"
                n["skipped_removed"] += 1
    elif skipped != "keep":
        raise ValueError(skipped)
    pending = []
    for rec, i in srcs:
        changed = rec["new_class"] != rec["original_class"]
        off = rec["offset_frames_4fps"]
        tgt = min(max(round_half_up(rec["new_time_s"]), lo), hi)
        if off and tgt != rec["original_second"]:
            ev = values[rec["new_class"]] if changed else {k: df.at[i, k] for k in EVENT_COLS}
            for k, v in values[idle].items():
                df.at[i, k] = v
            df.at[i, "revision"] = "moved_from"
            n["moved"] += 1
            if rec["new_class"] != idle:  # moving "no event" writes nothing
                pending.append((rec, tgt, ev, "both" if changed else "moved"))
            continue
        if changed:
            for k, v in values[rec["new_class"]].items():
                df.at[i, k] = v
        df.at[i, "revision"] = {(False, False): "none", (True, False): "class",
                                (False, True): "moved", (True, True): "both"}[changed, bool(off)]
        if times and rec["new_class"] != idle and not (confirmed_time == "raw" and rec["action"] == "confirmed"):
            df.at[i, "time_s"] = float(rec["new_time_s"])
        n["confirmed"] += not changed and not off
        n["class_changed"] += changed
        n["moved_same_second"] += bool(off)
    idle_gid = values[idle]["event_id"]
    for rec, tgt, ev, how in pending:
        at = [j for j in df.index[df["second"] == tgt] if df.at[j, "event_id"] == idle_gid]
        if at:
            j = at[0]
        else:  # the second already holds an event: add a row, as the original CSVs do
            j = df.index.max() + 1
            df.loc[j] = df.loc[df.index[df["second"] == tgt][0]]
            n["rows_added"] += 1
        for k, v in ev.items():
            df.at[j, k] = v
        df.at[j, "revision"] = how
        df.at[j, "orig_second"] = rec["original_second"]
        if times:
            df.at[j, "time_s"] = float(rec["new_time_s"])
    df = df.sort_values(["second", "frame_id"], kind="stable").reset_index(drop=True)
    return df, n


# ------------------------------------------------------------------- times, ignore, 4 Hz
def is_idle_name(name):
    return str(name).strip().lower() == "idle"


def ignore_mask(times, centers, ignore_sec):
    """
    1 where a row time is within ``ignore_sec`` of a skipped item's time.

    >>> ignore_mask(np.arange(10.0), [4.4], 2).tolist()
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    >>> ignore_mask(np.arange(4) / 4, [], 3).tolist()
    [0, 0, 0, 0]
    """
    times = np.asarray(times, dtype=float)
    out = np.zeros(len(times), dtype=int)
    for c in centers:
        out[np.abs(times - c) <= ignore_sec + 1e-9] = 1
    return out


def add_times_and_ignore(df, stamps, ignore_sec):
    """
    Fill ``time_s`` (see the module doc) and compute ``ignore`` in place on a revised 1 fps CSV.
    ``stamps``: `event_times.raw_stamps` of the patient (``{}``: no raw stamps, seconds are used).
    Returns the skipped items' times (centers of the ignore windows) and counts.
    """
    from event_times import stamp_for

    n = dict(stamps_matched=0, stamps_unmatched=0)
    centers = []
    if "orig_event_name" in df:
        for i in df.index[df["revision"] == "skipped"]:
            t = stamp_for(stamps, df.at[i, "orig_event_name"], df.at[i, "orig_second"])
            centers.append(float(df.at[i, "orig_second"] if t is None else t))
    if "time_s" in df:
        for i in df.index[df["time_s"].isna()]:
            if is_idle_name(df.at[i, "event_name"]):
                df.at[i, "time_s"] = float(df.at[i, "second"])
                continue
            t = stamp_for(stamps, df.at[i, "event_name"], df.at[i, "orig_second"])
            n["stamps_matched" if t is not None else "stamps_unmatched"] += 1
            df.at[i, "time_s"] = float(df.at[i, "second"] if t is None else t)
    if "orig_event_name" in df:
        df["ignore"] = ignore_mask(df["second"].to_numpy(), centers, ignore_sec)
    return centers, n


def to_4hz(df, centers, ignore_sec, idle_values, grid=4):
    """
    The 0.25 s grid version of a revised 1 fps CSV with ``time_s`` (see the module doc).

    >>> d = pd.DataFrame(dict(patient_id=1, patient_name="P", frame_id=[0, 60, 120], frame_path="x",
    ...     second=[0, 1, 2], event_id=[0, 5, 0], event_name=["Idle", "E", "Idle"], phase_id=1,
    ...     phase_name="ph", classes_id=[0, 2, 0], classes_name=["Idle", "g", "Idle"], time_s=[0, 1.6, 2]))
    >>> q = to_4hz(d, [], 3, dict(event_id=0, event_name="Idle", classes_id=0, classes_name="Idle"))
    >>> len(q), q.loc[q.event_name == "E", "frame_id"].tolist(), q.loc[q.event_name == "E", "second"].tolist()
    (12, [6], [1.5])
    >>> q.loc[q.frame_id == 4, "event_name"].item(), q.frame_path.iloc[0]
    ('Idle', 'P/frames_4fps.npy')
    """
    patient = df["patient_name"].iloc[0]
    base = df.drop_duplicates("second").sort_values("second").set_index("second")
    n_sec = len(base)
    assert (base.index.to_numpy() == np.arange(n_sec)).all(), f"{patient}: seconds not contiguous"
    k = np.arange(n_sec * grid)
    q = base.loc[k // grid].reset_index(drop=True)
    for col, v in idle_values.items():
        q[col] = v
    q["frame_id"] = k
    q["frame_path"] = f"{patient}/frames_{grid}fps.npy"
    q["second"] = k / grid
    q["time_s"] = k / grid
    q = q[list(df.columns)]
    for col in ("revision", "orig_event_name"):
        if col in q:
            q[col] = "none" if col == "revision" else ""
    if "orig_second" in q:
        q["orig_second"] = k // grid
    extra = []
    ev_rows = df[~df["event_name"].map(is_idle_name)]
    for _, r in ev_rows.iterrows():
        j = int(min(max(round_half_up(r["time_s"] * grid), 0), len(k) - 1))
        tgt = r.copy()
        tgt["frame_id"], tgt["frame_path"], tgt["second"] = j, q.at[j, "frame_path"], j / grid
        for col in ("phase_id", "phase_name"):
            tgt[col] = q.at[j, col]
        if is_idle_name(q.at[j, "event_name"]):
            for col in q.columns:
                q.at[j, col] = tgt[col]
        else:
            extra.append(tgt[q.columns])
    if extra:
        q = pd.concat([q, pd.DataFrame(extra)], ignore_index=True)
    if "ignore" in df:
        q["ignore"] = ignore_mask(q["second"].to_numpy(), centers, ignore_sec)
    q = q.sort_values(["frame_id"], kind="stable").reset_index(drop=True)
    q.attrs["rows_added"] = len(extra)
    return q


def class_of_rows(df, meta):
    """Local class index of each row (global id first, else the collapsed name), -1 if none."""
    gid = {g: c for c, g in enumerate(meta["class_global_ids"])}
    by_name = {collapse_key(nm): c for c, nm in enumerate(meta["class_csv_names"])}
    return [gid.get(int(e), by_name.get(collapse_key(nm), -1)) for e, nm in zip(df["event_id"], df["event_name"])]


def quantiles(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {}
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(n=int(len(x)), mean=float(x.mean()), p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4])


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from event_times import raw_stamps, stamp_for

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-dir", default=str(REVIEW_DIR), help="holds items.json and revisions.jsonl")
    ap.add_argument("--labels", default=str(LABEL_DIR), help="original RARPxx/Label/RARPxx_all_labels.csv")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--skipped", choices=["keep", "ignore"], default="keep",
                    help="keep: skips change nothing; ignore: event removed + ignore window")
    ap.add_argument("--ignore-sec", type=float, default=3.0, help="half width of the ignore window (s)")
    ap.add_argument("--times", action="store_true", help="add the fractional time_s column")
    ap.add_argument("--confirmed-time", choices=["review", "raw"], default="review",
                    help="time_s of confirmed items: the reviewed (integer) second or the raw stamp")
    ap.add_argument("--ann-dir", default=str(ANN_DIR), help="raw SURGICAL_PHASES json (for --times)")
    ap.add_argument("--out-4hz", help="also write the 0.25 s grid labels here (needs --times)")
    ap.add_argument("--report", help="aggregate report json")
    args = ap.parse_args()
    rev_dir, out = Path(args.review_dir), Path(args.out)
    for o in (out, Path(args.out_4hz) if args.out_4hz else None):
        if o is not None:
            assert o.resolve() != Path(args.labels).resolve(), "never overwrite the original labels"
    if args.out_4hz and not args.times:
        ap.error("--out-4hz needs --times")
    meta = json.loads((rev_dir / "items.json").read_text())
    meta["idle"] = [c for c, nm in enumerate(meta["names"]) if nm.lower() == "idle"][0]
    patients = sorted({it["patient"] for it in meta["items"]})
    dfs = {p: pd.read_csv(Path(args.labels) / p / "Label" / f"{p}_all_labels.csv") for p in patients}
    values = class_values(meta, events_lookup_of(dfs.values()))
    records = read_revisions(rev_dir / "revisions.jsonl")
    eff = effective_revisions(records)
    total = dict(lines=len(records), items_reviewed=len(eff),
                 skipped=sum(r["action"] == "skipped" for r in eff.values()))
    rep = dict(review_lines=len(records), items=len(meta["items"]), items_reviewed=len(eff),
               actions={a: sum(r["action"] == a for r in eff.values()) for a in ACTIONS},
               skipped_mode=args.skipped, ignore_sec=args.ignore_sec, confirmed_time=args.confirmed_time,
               patients=len(patients))
    per_class = {c: dict(name=nm, items=0, reviewed=0, **{a: 0 for a in ACTIONS}, changed_in=0, changed_out=0,
                         events_orig=0, events_revised=0) for c, nm in enumerate(meta["names"])}
    for it in meta["items"]:
        per_class[int(it["true"])]["items"] += 1
    for r in eff.values():
        pc = per_class[r["original_class"]]
        pc["reviewed"] += 1
        pc[r["action"]] += 1
        if r["action"] in ("changed", "changed+moved"):
            pc["changed_out"] += 1
            per_class[r["new_class"]]["changed_in"] += 1
    shifts, shifts_by_class, confirmed_vs_raw = [], {}, []
    agg4 = dict(rows=0, events=0, rows_added=0, ignore_rows=0, events_in_ignore=0)
    agg1 = dict(ignore_rows=0, events_in_ignore=0)
    for p in patients:
        recs = sorted((r for r in eff.values() if r["patient"] == p), key=lambda r: r["id"])
        new, n = revise_patient(dfs[p], recs, meta, values, skipped=args.skipped, times=args.times,
                                confirmed_time=args.confirmed_time)
        centers = []
        if args.times or args.skipped == "ignore":
            js = Path(args.ann_dir) / p / "SURGICAL_PHASES" / f"{p}.json"
            stamps = raw_stamps(js) if args.times else {}
            centers, n2 = add_times_and_ignore(new, stamps, args.ignore_sec)
            n.update(n2 if args.times else {})
            if args.times:  # shift = reviewed time - raw stamp of the source event (aggregate only)
                orig = dfs[p]
                for r in recs:
                    if r["action"] == "skipped" or r["original_class"] == meta["idle"]:
                        continue
                    i = find_source_row(orig, r, meta)
                    t0 = stamp_for(stamps, orig.at[i, "event_name"], r["original_second"])
                    if t0 is None:
                        continue
                    d = r["new_time_s"] - t0
                    if r["action"] in ("moved", "changed+moved"):
                        shifts.append(d)
                        shifts_by_class.setdefault(r["original_class"], []).append(d)
                    elif r["action"] == "confirmed":
                        confirmed_vs_raw.append(d)
        for k, v in n.items():
            total[k] = total.get(k, 0) + v
        if "ignore" in new:
            agg1["ignore_rows"] += int(new["ignore"].sum())
            agg1["events_in_ignore"] += int(((new["ignore"] == 1) & ~new["event_name"].map(is_idle_name)).sum())
        cls_o, cls_n = class_of_rows(dfs[p], meta), class_of_rows(new, meta)
        for c in cls_o:
            if c >= 0:
                per_class[c]["events_orig"] += 1
        for c in cls_n:
            if c >= 0:
                per_class[c]["events_revised"] += 1
        write_csv(new.drop(columns=["orig_event_name"], errors="ignore"), out / p / f"{p}_all_labels.csv")
        if args.out_4hz:
            q = to_4hz(new, centers, args.ignore_sec, values[meta["idle"]])
            agg4["rows"] += len(q)
            agg4["events"] += int((~q["event_name"].map(is_idle_name)).sum())
            agg4["rows_added"] += q.attrs["rows_added"]
            if "ignore" in q:
                agg4["ignore_rows"] += int(q["ignore"].sum())
                agg4["events_in_ignore"] += int(((q["ignore"] == 1) & ~q["event_name"].map(is_idle_name)).sum())
            write_csv(q.drop(columns=["orig_event_name"], errors="ignore"), Path(args.out_4hz) / p / f"{p}_all_labels.csv")
    print(json.dumps(total, indent=1))
    if args.report:
        rep["builder_counts"] = total
        rep["ignore_1fps"] = agg1
        if args.out_4hz:
            rep["grid_4hz"] = agg4
        rep["per_class"] = {nm["name"]: {k: v for k, v in nm.items() if k != "name"} for c, nm in per_class.items()}
        sh = np.asarray(shifts)
        rep["shift_moved_s"] = dict(signed=quantiles(sh), abs=quantiles(np.abs(sh)),
                                    hist_0_5s=np.histogram(sh, bins=np.arange(-8.25, 8.5, 0.5))[0].tolist(),
                                    hist_edges=[-8.25, 8.25, 0.5],
                                    integer_second_changed=int(sum(r["action"] in ("moved", "changed+moved")
                                                                   and round_half_up(r["new_time_s"]) != r["original_second"]
                                                                   for r in eff.values())))
        rep["shift_moved_by_class_s"] = {meta["names"][c]: quantiles(v) for c, v in sorted(shifts_by_class.items())}
        rep["confirmed_minus_raw_s"] = dict(signed=quantiles(confirmed_vs_raw), abs=quantiles(np.abs(confirmed_vs_raw)))
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=1, default=float))
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
