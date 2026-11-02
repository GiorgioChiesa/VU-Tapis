import math

import numpy as np

from tapis.evaluate import spotting_eval as se


def _gts():
    return {"v1": [(1, 100), (2, 200)], "v2": [(1, 50)]}


def test_perfect_predictions_give_map_one():
    preds = {v: [(c, t, 0.9) for c, t in evs] for v, evs in _gts().items()}
    res = se.spotting_map(preds, _gts(), deltas=(0, 1, 5), num_classes=3)
    assert res["map"] == {0: 1.0, 1: 1.0, 5: 1.0}
    assert res["n_gt"] == {1: 2, 2: 1}


def test_tolerance_controls_matching():
    preds = {v: [(c, t + 3, 0.9) for c, t in evs] for v, evs in _gts().items()}
    res = se.spotting_map(preds, _gts(), deltas=(1, 2, 3, 5), num_classes=3)
    assert res["map"][1] == 0.0 and res["map"][2] == 0.0
    assert res["map"][3] == 1.0 and res["map"][5] == 1.0


def test_duplicates_lower_ap_and_are_counted():
    gts = {"v": [(1, 100)]}
    preds = {"v": [(1, 100, 0.5), (1, 101, 0.9), (1, 102, 0.8)]}
    res = se.spotting_map(preds, gts, deltas=(5,), num_classes=2)
    # the highest score is matched first -> AP is still 1, duplicates are FPs after it
    assert res["map"][5] == 1.0
    op = se.operating_point(preds, gts, delta=5, durations={"v": 3600})
    assert op["tp"] == 1 and op["fp"] == 2
    assert op["duplicates_per_gt"] == 2.0
    assert op["false_alarms_per_hour"] == 2.0


def test_wrong_class_is_false_positive():
    gts = {"v": [(1, 10)]}
    preds = {"v": [(2, 10, 0.9)]}
    res = se.spotting_map(preds, gts, deltas=(1,), num_classes=3)
    assert res["ap"][1][1] == 0.0
    assert np.isnan(res["ap"][1][2])  # no ground truth for class 2
    assert res["map"][1] == 0.0


def test_decode_nms_vs_argmax():
    probs = np.zeros((20, 3))
    probs[:, 0] = 1.0
    probs[5:9] = [0.1, 0.8, 0.1]  # a 4-second plateau of class 1
    nms = se.decode_events(probs, nms_window=5, mode="nms", threshold=0.5)
    raw = se.decode_events(probs, mode="argmax")
    assert [(c, t) for c, t, _ in nms] == [(1, 5)]
    assert len(raw) == 4


def test_signed_error_and_average_map():
    gts = {"v": [(1, 10), (1, 40)]}
    preds = {"v": [(1, 12, 0.9), (1, 42, 0.8)]}
    op = se.operating_point(preds, gts, delta=5, durations={"v": 60})
    assert op["median_signed_error_s"] == 2.0 and op["mean_abs_error_s"] == 2.0
    res = se.spotting_map(preds, gts, deltas=(1, 2, 3), num_classes=2)
    assert math.isclose(se.average_map(res["map"], (1, 2, 3)), 2 / 3)
