"""Tests of the EXP-21 multi-task targets (no pytest needed: ``python tests/test_focus8.py``). Prints aggregates only."""

import contextlib
import doctest
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tools", ROOT / "detectron2"):
    sys.path.insert(0, str(p))

from tapis.datasets import focus8 as F  # noqa: E402

LAB4 = Path("/home/gchie/workspace/data/orsi_labels_revised/v2026-11-19_4hz")
ACT = Path("/home/gchie/workspace/nas_private/data/Orsi/action_labels")
S_RUN = Path("/home/gchie/workspace/wt/short-long-context/results/S/LONG/cv_ft/fold0")
TRAIN = ["RARP01", "RARP02", "RARP03", "RARP04", "RARP06", "RARP09", "RARP10", "RARP11", "RARP15", "RARP17"]


def test_doctests():
    assert doctest.testmod(F).failed == 0


def test_target_mapping_matches_orsi_collapse():
    """Per-target event counts of target_index == those of Orsi.collapse_event_dfs (the evaluation's classes)."""
    from eval_untrimmed import load_cfg

    from tapis.datasets.orsi import Orsi

    cfg = load_cfg(S_RUN, "checkpoint_best_steps.pyth")
    cfg.defrost()
    cfg.DATA.LABEL_DIR = str(LAB4)
    cfg.ENDOVIS_DATASET.TRAIN_LISTS = [f"{p}.csv" for p in TRAIN]
    cfg.ENDOVIS_DATASET.ADD_IDLE = 0
    with tempfile.TemporaryDirectory() as d:
        cfg.OUTPUT_DIR = d
        with contextlib.redirect_stdout(io.StringIO()):
            ds = Orsi(cfg, "train", load=False)
            ds._load_data()
    names = ds.filtered_dfs["event_name"].str.strip().str.replace(" ", "_")
    ours = np.zeros(len(F.TARGETS), int)
    for p in TRAIN:
        lab = F.VideoLabels(LAB4 / p / f"{p}_all_labels.csv")
        for t, k in lab.events:
            if k > 0 and not lab.ignore[min(int(round(t * 4)), lab.n - 1)]:
                ours[k - 1] += 1
    theirs = np.array([int((names.str.lower() == t.lower()).sum()) for t in F.TARGETS])
    assert (ours == theirs).all(), (ours.tolist(), theirs.tolist())
    assert ours.min() > 0


def test_event_targets():
    y, m = F.event_targets(200, [(10.1, 1), (10.6, 0), (30.0, 0), (40.0, 5)], fps=4, sigma_s=0.5,
                           nontarget_margin_s=1.0, ignore=np.r_[np.zeros(190, bool), np.ones(10, bool)])
    assert np.allclose(y.sum(1), 1)
    assert y[40, 1] > 0.9 and y[40].argmax() == 1          # 10.0 s, 0.1 s from the target
    assert m[40] and m[42]                                 # target rows beat the non-target mask
    assert not m[120] and not m[124] and m[125] and m[115]                # 30 s non-target masked +- 1 s only
    assert not m[195]                                      # ignore rows masked
    assert y[160, 5] == 1.0 and y[180, 0] == 1.0


def test_soft_ce_equals_ce_on_one_hot():
    logits = torch.randn(6, 9)
    tgt = torch.tensor([0, 1, 8, 3, 3, 0])
    w = torch.rand(9) + 0.5
    ref = torch.nn.functional.cross_entropy(logits, tgt, weight=w, reduction="none").mean()
    got = F.soft_ce(logits, torch.nn.functional.one_hot(tgt, 9).float(), torch.ones(6, dtype=torch.bool), w)
    assert torch.allclose(ref, got, atol=1e-5), (ref, got)
    half = F.soft_ce(logits, torch.nn.functional.one_hot(tgt, 9).float(), torch.tensor([1, 1, 1, 0, 0, 0]).bool())
    assert torch.allclose(half, torch.nn.functional.cross_entropy(logits[:3], tgt[:3]), atol=1e-5)


def test_asymmetric_bce():
    p, y = torch.rand(5, 7) * 0.98 + 0.01, torch.rand(5, 7)
    ref = torch.nn.functional.binary_cross_entropy(p, y)
    assert torch.allclose(F.asymmetric_bce(p, y, torch.ones(5, dtype=torch.bool)), ref, atol=1e-5)
    pw, nw = torch.ones(7), torch.ones(7)
    pw[5], nw[5] = 10.0, 0.1
    yy = torch.zeros(5, 7)
    lo = F.asymmetric_bce(p, yy, torch.ones(5, dtype=torch.bool), pw, nw)
    assert lo < F.asymmetric_bce(p, yy, torch.ones(5, dtype=torch.bool))  # negatives of class 5 down-weighted
    assert F.asymmetric_bce(p, y, torch.zeros(5, dtype=torch.bool)) == 0


def test_teacher_matches_orsi_signals_and_window():
    sys.path.insert(0, "/home/gchie/workspace/analisi/leonardo_action_sota/orsi")
    import orsi_signals

    sig, _, A, I = orsi_signals.load_signals("RARP01")
    act, ins = F.teacher_seconds(ACT / "RARP01_actions.json", len(sig[f"act:{A[0]}"]))
    ref_a = np.stack([sig[f"act:{a}"] for a in A], 1)
    ref_i = np.stack([sig[f"tins:{i}"] for i in I], 1)
    ok = np.isfinite(act).all(1)
    # orsi_signals writes 0 (not NaN) for seconds without regions: we mask them instead
    assert ok.sum() > 0.8 * len(ok)
    assert np.allclose(act[ok], ref_a[ok], atol=1e-6) and np.allclose(ins[ok], ref_i[ok], atol=1e-6)
    assert (ref_a[~ok & np.isfinite(ref_a).all(1)] == 0).all()
    rows = F.clip_rows(4 * 100, 4 * len(act))                      # clip centred at 100 s: seconds 98..101
    t, v = F.teacher_window(act, rows)
    secs = [s for s in range(98, 102) if ok[s]]
    assert v == bool(secs) and (not secs or np.allclose(t, act[secs].max(0)))


def test_clip_rows_match_get_sequence():
    from tapis.datasets import utils as du

    for c in (0, 3, 500, 999):
        assert list(F.clip_rows(c, 1000, 16)) == du.get_sequence(c, 16, 1, num_frames=1000, mode="center")


def test_sampling_weights_shares():
    cats = np.r_[np.full(1000, 0), np.full(50, 1), np.full(10, 2), np.full(30, -1), np.full(5, -2)]
    w = F.sampling_weights(cats, 0.5, 0.4, 0.1)
    assert abs(w.sum() - 1) < 1e-9 and w[cats == -2].sum() == 0
    assert abs(w[cats == 1].sum() - 0.25) < 1e-9 and abs(w[cats == 2].sum() - 0.25) < 1e-9
    assert abs(w[cats == 0].sum() - 0.4) < 1e-9 and abs(w[cats == -1].sum() - 0.1) < 1e-9


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if fails else 0)
