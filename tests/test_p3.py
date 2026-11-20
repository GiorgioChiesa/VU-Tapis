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
import clip_applier_split as CA  # noqa: E402
import online_eval_focus8 as OE  # noqa: E402
import confusion_focus8 as CF  # noqa: E402
import corr_focus8 as CO  # noqa: E402


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


def test_clip_applier_split():
    sig = np.zeros(100)
    sig[20] = 0.9  # a Clip Applier near t = 20
    flags = {"v": lambda t: CA.flag_at(sig, t, 2.0, 0.3)}
    gts = {"v": [(1, 21.0), (1, 60.0)]}
    dets = {"v": [(1, 21.0, 0.9), (1, 60.0, 0.4), (1, 80.0, 0.3)]}  # TP with, TP without, FP without
    r = CA.split_ap(dets, gts, flags, [1])[1]
    assert r["with"]["n_gt"] == 1 and r["with"]["2"] == 1.0
    assert r["without"]["n_gt"] == 1 and abs(r["without"]["2"] - 1.0) < 1e-9 and r["without"]["n_det"] == 2
    assert r["all"]["n_gt"] == 2 and r["all"]["n_det"] == 3


def test_online_decode_latency_and_phases():
    T = 80
    sec = np.arange(T) / 4
    probs = np.zeros((T, 3))
    probs[40, 1] = 0.9  # event of class 1 at t = 10 s
    em = OE.decode(probs, sec, [1, 2], "plain", None, 2.0)
    assert em == [(1, 10.0, 0.9, 12.0)], em  # emitted L = 2 s after the peak
    lat = OE.latency({"v": em}, {"v": [(1, 10.0)]}, 2.0, 2.0)
    assert lat["mean"] == 4.0 and lat["n_tp"] == 1  # 2 s decoder + 2 s clip delay
    p = np.tile(np.eye(3)[[0, 0, 1]].repeat(4, 0), (1, 1))  # 12 rows, 3 s
    last, secs = OE.last_row_per_second(p, np.arange(12) / 4)
    assert secs.tolist() == [0, 1, 2] and last.argmax(1).tolist() == [0, 0, 1]
    assert P.match_with_gt([10, 11, 30], [0.9, 0.8, 0.5], [10, 50], 2) == [(0, 0)]


def test_confusion_tables_and_plot():
    import tempfile

    names = ["bg"] + [f"e{i}" for i in range(8)]
    mat = np.zeros((10, 10), int)
    mat[1, 1], mat[2, 3], mat[9, 4], mat[3, 9] = 5, 2, 1, 1  # TP, confusion, false alarm row, missed column
    sub, labels = CF.table(mat, names)
    assert sub.shape == (9, 9) and sub[0, 0] == 5 and sub[1, 2] == 2 and sub[8, 3] == 1 and sub[2, 8] == 1
    with tempfile.TemporaryDirectory() as d:
        CF.plot(sub, labels + ["FA"], labels + ["M"], "t", Path(d) / "a.png")
        assert (Path(d) / "a.png").stat().st_size > 1000


def test_xcorr_recovers_the_lag_and_figures(tmpdir=None):
    import tempfile

    rng = np.random.default_rng(0)
    events, sigs = {}, {}
    for v in ("a", "b", "c"):
        ev_secs = sorted(rng.choice(np.arange(60, 540), 6, replace=False).tolist())
        s = rng.normal(size=(600, 2)) * 0.2
        for t in ev_secs:
            s[t + 7, 0] += 3.0  # signal 0 fires 7 s after every event of class 0
        events[v], sigs[v] = {0: ev_secs}, s
    r, p, n = CO.xcorr_events(events, sigs, 1, nshuf=50)
    lags = np.arange(-30, 31)
    assert lags[np.abs(r[0, 0]).argmax()] == 7 and p[0, 0] < 0.05 and abs(r[0, 1]).max() < abs(r[0, 0]).max()
    with tempfile.TemporaryDirectory() as d:
        m, lg = CO.heat(r, p, lags, ["e"], ["s0", "s1"], "t", Path(d) / "h.png")
        assert lg[0][0] == 7 and (Path(d) / "h.png").stat().st_size > 1000
        CO.lagged(r, lags, ["e"], ["s0", "s1"], "t", Path(d) / "l.png")


def test_doctests():
    for mod in (P, FU, CA, OE, CF, CO):
        r = doctest.testmod(mod)
        assert r.failed == 0, (mod.__name__, r)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
