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

Example (Vu-Tapis env):
    python tools/build_revised_labels.py
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path

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


def revise_patient(df, recs, meta, values):
    """
    Revised copy of one patient's CSV (``df`` untouched) and a dict of counts.
    ``recs``: effective revisions of this patient; ``values``: ``class_values``.
    Sources are resolved and cleared first, then moved events are written, so an event moved
    onto the second another event just left lands on that (now Idle) row.
    """
    df = df.copy()
    df["revision"] = "none"
    df["orig_second"] = df["second"].astype(int)
    idle = meta["idle"]
    lo, hi = int(df["second"].min()), int(df["second"].max())
    n = dict(confirmed=0, class_changed=0, moved=0, moved_same_second=0, rows_added=0)
    srcs = [(rec, find_source_row(df, rec, meta)) for rec in recs if rec["action"] != "skipped"]
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
    df = df.sort_values(["second", "frame_id"], kind="stable").reset_index(drop=True)
    return df, n


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-dir", default=str(REVIEW_DIR), help="holds items.json and revisions.jsonl")
    ap.add_argument("--labels", default=str(LABEL_DIR), help="original RARPxx/Label/RARPxx_all_labels.csv")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    rev_dir, out = Path(args.review_dir), Path(args.out)
    assert out.resolve() != Path(args.labels).resolve(), "never overwrite the original labels"
    meta = json.loads((rev_dir / "items.json").read_text())
    meta["idle"] = [c for c, nm in enumerate(meta["names"]) if nm.lower() == "idle"][0]
    patients = sorted({it["patient"] for it in meta["items"]})
    dfs = {p: pd.read_csv(Path(args.labels) / p / "Label" / f"{p}_all_labels.csv") for p in patients}
    values = class_values(meta, events_lookup_of(dfs.values()))
    records = read_revisions(rev_dir / "revisions.jsonl")
    eff = effective_revisions(records)
    total = dict(lines=len(records), items_reviewed=len(eff),
                 skipped=sum(r["action"] == "skipped" for r in eff.values()))
    for p in patients:
        recs = sorted((r for r in eff.values() if r["patient"] == p), key=lambda r: r["id"])
        new, n = revise_patient(dfs[p], recs, meta, values)
        for k, v in n.items():
            total[k] = total.get(k, 0) + v
        d = out / p
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{p}_all_labels.csv.tmp"
        new.to_csv(tmp, index=False)
        tmp.replace(d / f"{p}_all_labels.csv")
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    main()
