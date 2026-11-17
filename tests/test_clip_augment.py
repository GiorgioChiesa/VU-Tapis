"""Tests of the clip-consistent augmentation (no pytest needed: ``python tests/test_clip_augment.py``)."""

import doctest
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tapis.config.defaults import get_cfg  # noqa: E402
from tapis.datasets import clip_augment as ca  # noqa: E402


def all_on():
    aug = get_cfg().ORSI_AUG.clone()
    aug.defrost()
    for k in ("FLIP_P", "CROP_P", "ROT_P", "COLOR_P", "RAND_AUG_P", "BLUR_P", "SMOKE_P", "SPECULAR_P", "ERASE_P"):
        aug[k] = 1.0
    aug.ENABLE = True
    return aug


def test_doctests():
    assert doctest.testmod(ca).failed == 0


def test_identical_frames_stay_identical():
    """Same spatial/photometric params on every frame: a clip of copies stays a clip of copies."""
    aug = all_on()
    for seed in range(20):
        rng = random.Random(seed)
        frame = torch.rand(1, 3, 64, 80, generator=torch.Generator().manual_seed(seed))
        clip = frame.repeat(16, 1, 1, 1)
        p = ca.sample_params(aug, 64, 80, rng)
        out = ca.apply(clip, p)
        assert out.shape == clip.shape
        assert (out - out[:1]).abs().max() == 0, seed
        assert (out - clip).abs().max() > 0, seed


def test_each_frame_equals_single_frame_with_same_params():
    aug = all_on()
    clip = torch.rand(6, 3, 48, 48, generator=torch.Generator().manual_seed(1))
    for seed in range(10):
        p = ca.sample_params(aug, 48, 48, random.Random(seed))
        out = ca.apply(clip, p)
        for t in range(len(clip)):
            assert torch.allclose(out[t : t + 1], ca.apply(clip[t : t + 1], p), atol=1e-6), (seed, t)


def test_disabled_ops_are_identity():
    aug = get_cfg().ORSI_AUG.clone()
    aug.defrost()
    for k in ("FLIP_P", "CROP_P", "ROT_P", "COLOR_P", "RAND_AUG_P", "BLUR_P", "SMOKE_P", "SPECULAR_P", "ERASE_P"):
        aug[k] = 0.0
    clip = torch.rand(4, 3, 32, 32)
    p = ca.sample_params(aug, 32, 32, random.Random(0))
    assert torch.equal(ca.apply(clip, p), clip)


def test_temporal_shift_bounds():
    rng = random.Random(0)
    for rows in (1, 2):  # arm S: +/-1 row at 4 fps (0.25 s); arm L: +/-2 rows at 1 fps (2 s)
        s = [ca.temporal_shift(rows, rng) for _ in range(2000)]
        assert min(s) == -rows and max(s) == rows and set(s) == set(range(-rows, rows + 1))
    assert ca.temporal_shift(0, rng) == 0


def test_default_config_is_off():
    assert get_cfg().ORSI_AUG.ENABLE is False and get_cfg().ORSI_AUG.TEMPORAL_JITTER_ROWS == 0


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
