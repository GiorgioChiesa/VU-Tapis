"""
Tests of tools/prepare_external.py and of the jpg/static paths of tools/infer_untrimmed.py.
pytest is not installed: run them with

    python -c "import sys; sys.path[:0] = ['.', 'tests']; import test_prepare_external as t; \
        [getattr(t, f)() for f in dir(t) if f.startswith('test_')]"

The last test only reads the aggregate stats of the prepared datasets and is skipped
when they are missing.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "tools"), str(REPO)]

import prepare_external as pe  # noqa: E402


def test_gesture_mode_ties_and_gaps():
    fps = 60
    frames = [0, 6, 12, 18, 60, 66, 72, 78, 240, 246]
    labels = [3, 3, 5, 5, 1, 2, 2, 2, 7, 7]
    g = pe.gesture_per_second(frames, labels, n_seconds=4, fps=fps)
    # second 0: tie 3/5 -> smallest id; 1: mode 2; 2-3: no samples; second 4 is beyond n_seconds
    assert g.tolist() == [3, 2, -1, -1]
    assert g.dtype == np.int16


def test_gesture_boundary_belongs_to_next_second():
    g = pe.gesture_per_second([54, 60], [1, 4], n_seconds=2)
    assert g.tolist() == [1, 4]


def test_multi_hot_offsets():
    mh = pe.multi_hot([[1, 3], [], [14]], 14, 1)
    assert mh.shape == (3, 14) and mh.dtype == np.uint8
    assert mh[0].nonzero()[0].tolist() == [0, 2]
    assert mh[1].sum() == 0
    assert mh[2].nonzero()[0].tolist() == [13]
    assert pe.multi_hot([[0, 0, 20]], 21, 0).sum() == 2  # repeated classes count once


def test_read_yolo_empty_and_instances():
    with tempfile.TemporaryDirectory() as d:
        empty, full = Path(d) / "a.txt", Path(d) / "b.txt"
        empty.write_text("")
        full.write_text("3 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n20 0.1 0.1 0.1 0.1\n\n")
        assert pe.read_yolo(empty, 21) == []
        assert pe.read_yolo(full, 21) == [3, 3, 20]


def test_load_jpg_matches_tapis_val_transform():
    import cv2

    import infer_untrimmed as iu

    h, w = 300, 480
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 2] = 200  # red in BGR
    img[:, w // 2 :, 0] = 100  # right half also blue
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"  # lossless, so values are exact
        cv2.imwrite(str(p), img)
        t = iu.load_jpg(p)
    assert t.shape == (3, 224, 224) and str(t.dtype) == "torch.uint8"
    assert (t[0] == 200).all()  # channel 0 is R after BGR -> RGB
    assert (t[1] == 0).all()
    # short side 300 -> 224, long side floor(480 * 224 / 300) = 358, center crop starts at ceil(134 / 2) = 67
    left, right = t[2, :, :100], t[2, :, -100:]
    assert (left == 0).all() and (right == 100).all()


def test_static_clips_repeat_the_center_frame():
    import infer_untrimmed as iu

    cfg = SimpleNamespace(DATA=SimpleNamespace(NUM_FRAMES=16, SAMPLING_RATE=1, SEQ_MODE="center"))
    dyn = iu.clip_indices(40, cfg)
    sta = iu.clip_indices(40, cfg, static=True)
    assert dyn.shape == sta.shape == (40, 16)
    assert (sta == np.arange(40)[:, None]).all()
    assert dyn[20].tolist() == list(range(12, 28))  # center mode: i-8 .. i+7
    assert dyn[0].min() == 0 and dyn[-1].max() == 39  # clamped at the borders


def test_prepared_stats_match_inventory():
    root = pe.OUT_DIR
    metas = {ds: root / d / "meta.json" for ds, d in (("grasp", "grasp"), ("sar", "sar_rarp50"), ("esad", "esad"))}
    if not all(p.exists() for p in metas.values()):
        print("skip: prepared datasets not found")
        return
    m = {ds: json.loads(p.read_text()) for ds, p in metas.items()}
    for split in ("train", "test"):
        st = m["grasp"]["stats"][split]
        assert st["annotated_seconds"] == pe.EXPECTED["grasp"]["seconds"][split]
        assert st["keyframes"] == pe.EXPECTED["grasp"]["keyframes"][split]
    rows = sum(m["sar"]["stats"][s]["rows_10hz_all_videos"] for s in ("train", "test"))
    assert rows == pe.EXPECTED["sar"]["rows_10hz"]
    for split, n in pe.EXPECTED["esad"]["instances"].items():
        assert m["esad"]["stats"][split]["instances"] == n
    for ds in m.values():
        for video in (v for s in ds["splits"].values() for v in s):
            assert video not in ds["excluded"]
