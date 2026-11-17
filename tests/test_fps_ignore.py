"""Tests of the frame-rate conversion and ignore windows (no pytest needed: ``python tests/test_fps_ignore.py``)."""

import doctest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tools", ROOT / "detectron2"):
    sys.path.insert(0, str(p))

from tapis.evaluate import spotting_eval as se  # noqa: E402

import eval_untrimmed as eu  # noqa: E402
import train_temporal as tt  # noqa: E402


def test_doctests():
    for m in (se, tt):
        assert doctest.testmod(m).failed == 0, m.__name__


def upsample(curve, fps):
    """1 Hz curve -> fps Hz curve (each second repeated, peak kept on the first row)."""
    out = np.repeat(curve, fps, axis=0) * 0.5
    out[::fps] = curve
    return out


def decode_seconds(probs, fps, nms_s=5.0):
    return [(c, t / fps, s) for c, t, s in se.decode_events(probs, se.window_frames(nms_s, fps), 0.0, (0,))]


def test_nms_window_is_in_seconds():
    p1 = np.zeros((40, 2))
    p1[[10, 13, 25], 1] = [0.9, 0.8, 0.7]  # 13 s is within 5 s of 10 s, 25 s is not
    p1[:, 0] = 1 - p1[:, 1]
    d1 = decode_seconds(p1, 1)
    d4 = decode_seconds(upsample(p1, 4), 4)
    assert [t for _, t, _ in d1] == [10.0, 25.0]
    assert [t for _, t, _ in d4 if _ > 0.6] == [10.0, 25.0]


def test_tolerance_is_in_seconds_at_4hz():
    gts = {"v": [(1, 10.0)]}
    for t_pred, delta, tp in [(10.75, 1, True), (11.25, 1, False), (11.25, 2, True), (10.125, 1, True)]:
        res = se.spotting_map({"v": [(1, t_pred, 0.9)]}, gts, (delta,), num_classes=2, ignore_classes=(0,))
        assert (res["map"][delta] == 1.0) == tp, (t_pred, delta)


def test_ignore_windows_1hz_and_4hz_cover_the_same_time():
    i1 = se.ignore_intervals(np.arange(7, 14), 1)  # rows 7..13 s (+/-3 s around 10)
    i4 = se.ignore_intervals(np.arange(28, 53) / 4, 4)  # rows 7..13 s on the 0.25 s grid
    assert i1 == [(6.5, 13.5)] and i4 == [(6.875, 13.125)]
    dets = [(1, t, 0.5) for t in (6.0, 7.0, 10.0, 13.0, 14.0)]
    assert [t for _, t, _ in se.drop_ignored(dets, i1)] == [6.0, 14.0]
    assert [t for _, t, _ in se.drop_ignored(dets, i4)] == [6.0, 14.0]


def test_ignore_rows_mask_loss_rows():
    m4 = tt.ignore_rows(80, se.ignore_intervals(np.arange(28, 53) / 4, 4), 4)
    assert np.flatnonzero(m4).tolist() == list(range(28, 53))
    m1 = tt.ignore_rows(20, se.ignore_intervals(np.arange(7, 14), 1), 1)
    assert np.flatnonzero(m1).tolist() == list(range(7, 14))


def test_targets_rows_from_seconds():
    fps, k_rows = 4, int(round(0.75 * 4))
    y, _ = tt.event_targets(80, [(1, 10.3 * fps)], 2, "dilate", k_rows)
    assert np.flatnonzero(y[:, 1]).tolist() == list(range(38, 45))  # row 41 (10.25 s) +/- 3 rows
    y1, _ = tt.event_targets(20, [(1, 10.3)], 2, "dilate", 3)
    assert np.flatnonzero(y1[:, 1]).tolist() == list(range(7, 14))


def test_class_groups_restrict_metrics():
    names = ["Idle", "A", "B", "C"]
    groups = eu.parse_class_groups(["short:A,B", "C"], names)
    assert groups == {"short": [1, 2], "C": [3]}
    gts = {"v": [(1, 5.0), (2, 20.0), (3, 40.0)]}
    dets = {"v": [(1, 5.0, 0.9), (2, 30.0, 0.8), (3, 40.0, 0.7)]}
    out = eu.subset_metrics(dets, gts, groups, names, (0,), {"v": 100.0}, deltas=(1,))
    assert out["short"]["map"]["1"] == 0.5 and out["C"]["map"]["1"] == 1.0
    assert set(out["short"]["n_gt"]) == {"A", "B"}
    try:
        eu.parse_class_groups(["X"], names)
        raise AssertionError("unknown class accepted")
    except ValueError:
        pass


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
