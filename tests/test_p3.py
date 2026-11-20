"""Tests of the EXP-21 phase 3 tools on synthetic arrays (no pytest, no patient data): ``python tests/test_p3.py``."""

import doctest
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _n, _sub in (("tapis", ""), ("tapis.datasets", "datasets"), ("tapis.evaluate", "evaluate")):  # skip tapis/__init__
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        _m.__path__ = [str(ROOT / "tapis" / _sub)] if _sub else [str(ROOT / "tapis")]
        sys.modules[_n] = _m
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]

import p3lib as P  # noqa: E402
import fuse_focus8 as FU  # noqa: E402


def test_equal_weights_give_the_plain_mean():
    rng = np.random.default_rng(0)
    a, b = rng.random((50, 5)), rng.random((50, 5))
    mean = (a + b) / 2
    assert np.allclose(P.fuse_global(a, b, 0.5), mean)
    c = rng.random(50)
    assert np.allclose(P.fuse_conf(a, b, c, c), mean)  # equal confidences
    assert np.allclose(P.fuse_conf(a, b, np.zeros(50), np.zeros(50)), mean)  # both zero: plain mean too
    assert np.allclose(P.fuse_global(a, b, 1.0), a) and np.allclose(P.fuse_global(a, b, 0.0), b)


def test_conf_weight_follows_confidence():
    a, b = np.array([[0.0, 1.0]]), np.array([[0.0, 0.0]])
    out = P.fuse_conf(a, b, np.array([0.9]), np.array([0.1]))
    assert np.allclose(out, 0.9 * a + 0.1 * b)


def test_fuse_videos_keys_and_rows():
    sec = np.arange(8) / 4
    A = {"v": {"second": sec, "probs_steps": np.full((8, 5), 0.2), "probs_phases": np.full((8, 3), 0.5)}}
    B = {"v": {"second": sec, "probs_steps": np.full((8, 5), 0.4), "probs_phases": np.full((8, 3), 0.1)}}
    e = P.fuse_videos(A, B, "global", True, 0.5)["v"]
    assert set(e) == {"second", "probs_steps"} and np.allclose(e["probs_steps"], 0.3)
    p = P.fuse_videos(A, B, "conf", False)["v"]
    assert set(p) == {"second", "probs_phases"} and np.allclose(p["probs_phases"], 0.5 * 0.5 / 0.6 + 0.1 * 0.1 / 0.6)
    B["v"]["second"] = sec + 1
    try:
        P.fuse_videos(A, B, "global", True, 0.5)
        raise SystemExit("rows mismatch not detected")
    except AssertionError:
        pass


def test_wilcoxon_and_nan_handling():
    x = {f"v{i}": 0.5 + 0.1 * i for i in range(8)}
    y = {k: v - 0.2 for k, v in x.items()}
    r = P.wilcoxon(x, y)
    assert r["n_videos"] == 8 and r["n_better"] == 8 and r["p"] < 0.01
    assert P.wilcoxon(x, x)["p"] == 1.0
    y["v0"] = float("nan")
    assert P.wilcoxon(x, y)["n_videos"] == 7


def test_eval_events_per_video_and_selection():
    """Perfect detections give AP 1 per video; shifted ones lose at +-1 s."""
    names = P.LONG_NAMES
    gts = {"a": [(1, 10.0), (2, 50.0)], "b": [(1, 20.0)]}
    dets = {"a": [(1, 10.0, 0.9), (2, 53.0, 0.8)], "b": [(1, 20.0, 0.7)]}  # class 2 off by 3 s
    res = P.eval_events(dets, gts, names, ["a", "b"])
    g = res["group_map"]["long"]
    assert abs(g["1"] - 0.5) < 1e-9 and abs(g["2"] - 0.5) < 1e-9 and abs(g["5"] - 1.0) < 1e-9, g
    pv = res["per_video_group_ap"]["long"]
    assert pv["1"]["a"] == 0.5 and pv["1"]["b"] == 1.0 and pv["5"]["a"] == 1.0


def test_thresholds_and_dump_json(tmpdir=None):
    import json
    import tempfile

    th = P.best_thresholds({"v": [(1, 10.0, 0.9), (1, 50.0, 0.3)]}, {"v": [(1, 10.5)]}, [1, 2])
    assert th == {1: 0.35, 2: 0.5}
    kept = P.apply_thresholds({"v": [(1, 1.0, 0.9), (1, 2.0, 0.3)]}, th)
    assert kept == {"v": [(1, 1.0, 0.9)]}
    with tempfile.TemporaryDirectory() as d:
        P.dump_json({"x": np.float32("nan"), "y": [np.int64(3), 1.5]}, Path(d) / "a.json")
        assert json.loads((Path(d) / "a.json").read_text()) == {"x": None, "y": [3, 1.5]}


def test_doctests():
    for mod in (P, FU):
        r = doctest.testmod(mod)
        assert r.failed == 0, (mod.__name__, r)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
