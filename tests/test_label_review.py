"""Tests of the label-review tools (no pytest needed: ``python tests/test_label_review.py``)."""

import doctest
import json
import multiprocessing as mp
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import build_revised_labels as brl  # noqa: E402
import export_label_review as elr  # noqa: E402
import serve_viewer as sv  # noqa: E402

NAMES = ["Idle", "A", "B"]
META = dict(names=NAMES, class_global_ids=[0, 11, 12], class_csv_names=["Idle", "Event_A", "Event_B"], idle=0)


def test_doctests():
    assert doctest.testmod(elr).failed == 0
    assert doctest.testmod(brl).failed == 0


# ---------------------------------------------------------------------------- key map
def test_key_map_qwerty_then_digits():
    km = elr.key_map([f"c{i}" for i in range(33)])
    assert [km[i] for i in range(10)] == list("qwertyuiop")
    assert km[10] == "a" and km[19] == "z" and km[25] == "m"
    assert [km[i] for i in range(26, 33)] == list("1234567")
    assert len(set(km.values())) == 33


def test_key_map_too_many_classes():
    try:
        elr.key_map([f"c{i}" for i in range(34)])
    except ValueError:
        return
    raise AssertionError("34 classes must not fit 33 keys")


# ------------------------------------------------------------------------- 4 fps grid
def test_window_times_grid():
    w = elr.window_times(100, 3600.0)
    assert len(w) == 64 == elr.N_FRAMES
    assert w[elr.KEY_INDEX] == 100.0
    assert w[0] == 92.0 and w[-1] == 107.75
    assert all(abs((b - a) - 0.25) < 1e-9 for a, b in zip(w, w[1:]))


def test_window_times_clamped():
    w = elr.window_times(3, 3600.0)
    assert w[:21] == [0.0] * 21 and w[21] == 0.25 and w[elr.KEY_INDEX] == 3.0
    w = elr.window_times(3598, 3599.5)
    assert w[elr.KEY_INDEX] == 3598.0 and max(w) == 3599.5 and w[-1] == 3599.5
    assert w == sorted(w)


# ----------------------------------------------------------------- revised CSV builder
def toy_csv():
    """Seconds 0..20, Idle everywhere, event A at 5 and event B at 10."""
    rows = []
    for s in range(21):
        ev = {5: (11, "Event_A", 2, "grpA"), 10: (12, "Event_B", 3, "grpB")}.get(s, (0, "Idle", 0, "Idle"))
        rows.append(dict(patient_id=1, patient_name="P", frame_id=60 * s, frame_path=f"P/{s}.jpg", second=s,
                         event_id=ev[0], event_name=ev[1], phase_id=1, phase_name="ph",
                         classes_id=ev[2], classes_name=ev[3]))
    return pd.DataFrame(rows)


def item(second, cls):
    return dict(id=f"P_t{second:05d}_{'e' if cls else 'c'}{cls:02d}", patient="P", second=second, true=cls)


def rec(second, cls, new, off, skipped=False):
    return brl.make_record(item(second, cls), new, off, skipped, len(NAMES))


def build(records):
    df = toy_csv()
    values = brl.class_values(META, brl.events_lookup_of([df]))
    eff = brl.effective_revisions(records)
    new, n = brl.revise_patient(df, sorted(eff.values(), key=lambda r: r["id"]), META, values)
    assert df.equals(toy_csv()), "the original frame must stay untouched"
    return new, n


def ev(df, s):
    return list(df.loc[df["second"] == s, "event_name"])


def test_move_event():
    new, n = build([rec(5, 1, 1, +8)])  # +8 frames = +2 s
    assert ev(new, 5) == ["Idle"] and ev(new, 7) == ["Event_A"]
    assert new.loc[new.second == 7, "revision"].item() == "moved"
    assert new.loc[new.second == 7, "orig_second"].item() == 5
    assert new.loc[new.second == 7, "classes_id"].item() == 2
    assert new.loc[new.second == 5, "revision"].item() == "moved_from"
    assert new.loc[new.second == 5, "classes_name"].item() == "Idle"
    assert n["moved"] == 1 and len(new) == 21


def test_small_move_rounds_to_same_second():
    new, _ = build([rec(5, 1, 1, +1)])  # 5.25 s -> row 5
    assert ev(new, 5) == ["Event_A"] and new.loc[new.second == 5, "revision"].item() == "moved"
    new, _ = build([rec(5, 1, 1, +2)])  # 5.5 s rounds half up -> row 6
    assert ev(new, 6) == ["Event_A"] and ev(new, 5) == ["Idle"]


def test_change_class_only_touches_that_row():
    new, n = build([rec(10, 2, 1, 0)])
    assert ev(new, 10) == ["Event_A"]
    r = new.loc[new.second == 10].iloc[0]
    assert r.event_id == 11 and r.classes_id == 2 and r.revision == "class"
    orig = toy_csv()
    diff = (new[orig.columns] != orig).any(axis=1)
    assert list(new.loc[diff, "second"]) == [10]
    assert n["class_changed"] == 1


def test_change_and_move_onto_occupied_second_adds_row():
    new, n = build([rec(5, 1, 2, +20)])  # 5 s + 5 s = 10 s, which holds event B
    assert sorted(ev(new, 10)) == ["Event_B", "Event_B"] and ev(new, 5) == ["Idle"]
    assert n["rows_added"] == 1 and len(new) == 22
    assert set(new.loc[new.second == 10, "revision"]) == {"none", "both"}


def test_control_to_event_and_event_to_idle():
    new, _ = build([rec(15, 0, 2, 0), rec(10, 2, 0, 0)])
    assert ev(new, 15) == ["Event_B"] and ev(new, 10) == ["Idle"]


def test_source_row_of_collapsed_class_in_shared_second():
    df = toy_csv()
    extra = df[df.second == 10].copy()
    extra[["event_id", "event_name", "classes_id", "classes_name"]] = [99, "Event left_A", 4, "grpL"]
    df2 = pd.concat([df, extra], ignore_index=True)  # second 10: raw B + a raw event that collapses to A
    meta = dict(META, class_csv_names=["Idle", "Event right_A", "Event_B"])
    i = brl.find_source_row(df2, rec(10, 1, 1, 0), meta)
    assert df2.at[i, "event_id"] == 99
    i = brl.find_source_row(df2, rec(10, 2, 2, 0), meta)
    assert df2.at[i, "event_id"] == 12


def test_skip_changes_nothing():
    new, n = build([rec(5, 1, 2, +4, skipped=True)])
    orig = toy_csv()
    assert (new[orig.columns] == orig).all().all() and set(new.revision) == {"none"}
    assert n == dict(confirmed=0, class_changed=0, moved=0, moved_same_second=0, rows_added=0)


def test_last_wins_and_skip_does_not_undo():
    rs = [rec(5, 1, 2, 0), rec(5, 1, 1, +8), rec(5, 1, 1, 0, skipped=True)]
    eff = brl.effective_revisions(rs)
    assert eff[item(5, 1)["id"]]["action"] == "moved"
    new, _ = build(rs)
    assert ev(new, 7) == ["Event_A"] and ev(new, 5) == ["Idle"]
    new, _ = build(rs[:1])
    assert ev(new, 5) == ["Event_B"]


def build_ext(records, stamps=None, ignore_sec=3):
    df = toy_csv()
    values = brl.class_values(META, brl.events_lookup_of([df]))
    eff = brl.effective_revisions(records)
    new, n = brl.revise_patient(df, sorted(eff.values(), key=lambda r: r["id"]), META, values,
                                skipped="ignore", times=True)
    centers, n2 = brl.add_times_and_ignore(new, stamps or {}, ignore_sec)
    return new, centers, values, {**n, **n2}


def test_skip_ignore_removes_event_and_masks_window():
    new, centers, _, n = build_ext([rec(5, 1, 1, 0, skipped=True)], stamps={"Event_A": [5.3], "Event_B": [9.8]})
    assert ev(new, 5) == ["Idle"] and new.loc[new.second == 5, "revision"].item() == "skipped"
    assert centers == [5.3] and n["skipped_removed"] == 1
    assert list(new.loc[new.ignore == 1, "second"]) == [3, 4, 5, 6, 7, 8]
    assert new.loc[new.second == 10, "time_s"].item() == 9.8 and n["stamps_matched"] == 1


def test_times_of_reviewed_rows_and_4hz_grid():
    new, centers, values, _ = build_ext([rec(5, 1, 1, +3)])  # 5.75 s, stays on row 6 at 1 fps
    assert ev(new, 6) == ["Event_A"] and new.loc[new.second == 6, "time_s"].item() == 5.75
    assert new.loc[new.second == 10, "time_s"].item() == 10.0  # no raw stamp: the second
    q = brl.to_4hz(new, centers, 3, values[0])
    assert len(q) == 84 and list(q.columns) == list(new.columns)
    assert q.loc[q.event_name == "Event_A", "frame_id"].tolist() == [23]
    assert q.loc[q.event_name == "Event_B", "frame_id"].tolist() == [40]
    assert (q["ignore"] == 0).all() and q["frame_id"].is_monotonic_increasing


def test_4hz_collision_adds_row_and_ignore_on_grid():
    new, centers, values, _ = build_ext([rec(5, 1, 1, +20), rec(15, 0, 0, 0, skipped=True)])  # A onto B's 10 s
    q = brl.to_4hz(new, centers, 1, values[0])
    assert sorted(q.loc[q.frame_id == 40, "event_name"]) == ["Event_A", "Event_B"] and q.attrs["rows_added"] == 1
    assert q.loc[q.ignore == 1, "second"].tolist() == [14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0]


def test_make_record_validation():
    for bad in [(1, 99, 0), (1, 1, 32), (1, 1, -33)]:
        try:
            rec(5, bad[0], bad[1], bad[2])
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    r = rec(5, 1, 1, -4)
    assert r["new_time_s"] == 4.0 and r["action"] == "moved"
    assert rec(1, 1, 1, -32)["new_time_s"] == 0.0


# ------------------------------------------------------------------- concurrent append
def _writer(path, k, n):
    for i in range(n):
        brl.append_revision(path, dict(id=f"w{k}_{i}", action="confirmed", pad="x" * 5000))


def test_concurrent_append():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "revisions.jsonl"
        ctx = mp.get_context("fork")
        ps = [ctx.Process(target=_writer, args=(str(path), k, 60)) for k in range(8)]
        [p.start() for p in ps]
        [p.join() for p in ps]
        assert all(p.exitcode == 0 for p in ps)
        recs = brl.read_revisions(path)
        assert len(recs) == 480 and len({r["id"] for r in recs}) == 480
        assert len(path.read_text().splitlines()) == 480


def test_torn_last_line_ignored():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.jsonl"
        brl.append_revision(path, dict(id="a", action="confirmed"))
        with open(path, "a") as f:
            f.write('{"id": "b", "act')
        assert [r["id"] for r in brl.read_revisions(path)] == ["a"]


# ---------------------------------------------------------------------------------- api
def test_api_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        it = dict(item(5, 1), fold=0, type="event", frames="P_t00005")
        meta = dict(names=NAMES, keys=["q", "w", "e"], n_frames=64, key_index=32, fps=4, items=[it])
        Path(d, "items.json").write_text(json.dumps(meta))
        sv.ReviewHandler.meta, sv.ReviewHandler.items = meta, {it["id"]: it}
        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(sv.ReviewHandler, directory=d))
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def post(body):
            req = urllib.request.Request(base + "/api/review", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        try:
            assert post(dict(id=it["id"], new_class=2, offset_frames_4fps=-3))[0] == 200
            assert post(dict(id=it["id"], skipped=True))[0] == 200
            assert post(dict(id="nope", new_class=1))[0] == 400
            assert post(dict(id=it["id"], new_class=1, offset_frames_4fps=40))[0] == 400
            with urllib.request.urlopen(base + "/api/items") as r:
                got = json.loads(r.read())
            rv = got["reviews"][it["id"]]
            assert rv["action"] == "changed+moved" and rv["new_time_s"] == 4.25 and rv["new_class"] == 2
            assert len(brl.read_revisions(Path(d) / "revisions.jsonl")) == 2
            with urllib.request.urlopen(base + "/") as r:
                assert b"label_review.js" in r.read()
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
