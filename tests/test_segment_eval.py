import doctest

import numpy as np

from tapis.evaluate import segment_eval as sg


def test_doctests():
    assert doctest.testmod(sg).failed == 0


def test_perfect_prediction():
    gt = np.repeat([0, 3, 1, 4], [50, 20, 30, 10])
    m = sg.segmental_metrics({"a": gt, "b": gt[::-1]}, {"a": gt, "b": gt[::-1]})
    assert m == {"edit": 100.0, "F1@10": 100.0, "F1@25": 100.0, "F1@50": 100.0}


def test_flicker_hurts_segments_not_accuracy():
    gt = np.repeat([0, 1], [100, 100])
    pred = gt.copy()
    pred[[20, 40, 60]] = 1  # 99% frame accuracy, 7 predicted segments instead of 2
    m = sg.segmental_metrics({"v": pred}, {"v": gt})
    assert (pred == gt).mean() > 0.98
    assert m["edit"] < 50 and m["F1@50"] < 50


def test_shifted_boundary_depends_on_overlap():
    gt = np.repeat([0, 1], [100, 100])
    pred = np.repeat([0, 1], [140, 60])  # class 1 IoU = 0.6, class 0 IoU = 100/140
    m = sg.segmental_metrics({"v": pred}, {"v": gt}, overlaps=(50, 75))
    assert m["F1@50"] == 100.0 and m["F1@75"] == 0.0
    assert m["edit"] == 100.0


def test_counts_sum_over_videos():
    gt = {"a": [0, 0, 1, 1], "b": [2, 2, 2, 2]}
    pred = {"a": [0, 0, 1, 1], "b": [3, 3, 3, 3]}
    m = sg.segmental_metrics(pred, gt)
    # tp = 2, fp = 1, fn = 1 -> P = R = 2/3
    assert abs(m["F1@50"] - 200 / 3) < 1e-9
    assert m["edit"] == 50.0


def test_ignore_classes():
    gt = [0, 1, 1, 0, 0, 2]
    pred = [1, 1, 1, 0, 2, 2]
    assert sg.edit_score(pred, gt, ignore=(0,)) == 100.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
