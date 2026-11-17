"""
Clip-consistent training augmentation for the Orsi frame dataset (``cfg.ORSI_AUG``).

Every random parameter is drawn once per clip (`sample_params`) and the same transform is then
applied to all frames (`apply`): the whole clip is one ``[T, C, H, W]`` tensor and every operation
acts on it as a batch, with one shared parameter set. Input and output are float tensors in [0, 1]
(the value range of the ``Video_1fps`` tensors; no mean/std normalisation, as in the current
training and inference). Channel order is left as stored (BGR); the colour operations used here
are channel-symmetric except saturation and RandAugment's Color/Posterize-like ops, whose luma
weights then apply to swapped channels.

Operations (each with its own probability / range in the config):
  flip (horizontal), scale jitter + random crop back to the input size, rotation, colour jitter
  (brightness / contrast / saturation), RandAugment, gaussian blur, synthetic smoke (blurred noise
  veil + contrast drop), specular highlights (small saturated white ellipses), random erasing.
"""

import math
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def parse_rand_augment(spec):
    """
    ``"rand-m7-n4"`` (timm style, other fields ignored) -> (magnitude, num_ops); "" -> None.

    >>> parse_rand_augment("rand-m7-n4-mstd0.5-inc1"), parse_rand_augment("")
    ((7, 4), None)
    """
    if not spec:
        return None
    m, n = 9, 2
    for part in spec.split("-")[1:]:
        if part.startswith("m") and part[1:].isdigit():
            m = int(part[1:])
        elif part.startswith("n") and part[1:].isdigit():
            n = int(part[1:])
    return m, n


def sample_params(aug, height, width, rng=random):
    """One parameter set for a whole clip (plain Python values, loggable)."""
    p = {}
    p["flip"] = rng.random() < aug.FLIP_P
    if rng.random() < aug.CROP_P:
        area = rng.uniform(*aug.CROP_SCALE)
        side = math.sqrt(area)
        h, w = max(8, int(round(height * side))), max(8, int(round(width * side)))
        p["crop"] = (rng.randint(0, height - h), rng.randint(0, width - w), h, w)
    p["rotate"] = rng.uniform(-aug.ROT_DEG, aug.ROT_DEG) if rng.random() < aug.ROT_P else 0.0
    if rng.random() < aug.COLOR_P:
        p["color"] = [rng.uniform(1 - aug.BRIGHTNESS, 1 + aug.BRIGHTNESS),
                      rng.uniform(1 - aug.CONTRAST, 1 + aug.CONTRAST),
                      rng.uniform(1 - aug.SATURATION, 1 + aug.SATURATION)]
    ra = parse_rand_augment(aug.RAND_AUG)
    if ra and rng.random() < aug.RAND_AUG_P:
        p["rand_aug"] = {"magnitude": ra[0], "num_ops": ra[1], "seed": rng.randrange(2**31)}
    if rng.random() < aug.BLUR_P:
        p["blur_sigma"] = rng.uniform(*aug.BLUR_SIGMA)
    if rng.random() < aug.SMOKE_P:
        p["smoke"] = {"alpha": rng.uniform(*aug.SMOKE_ALPHA), "grey": rng.uniform(0.75, 1.0),
                      "contrast": rng.uniform(*aug.SMOKE_CONTRAST), "seed": rng.randrange(2**31)}
    if rng.random() < aug.SPECULAR_P:
        n = rng.randint(*aug.SPECULAR_N)
        p["specular"] = [(rng.uniform(0, height), rng.uniform(0, width), rng.uniform(2, 10), rng.uniform(2, 10),
                          rng.uniform(0, math.pi)) for _ in range(n)]
    if rng.random() < aug.ERASE_P:
        area = rng.uniform(0.02, 0.2) * height * width
        ratio = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
        h = min(height, int(round(math.sqrt(area * ratio))))
        w = min(width, int(round(math.sqrt(area / ratio))))
        p["erase"] = (rng.randint(0, height - h), rng.randint(0, width - w), h, w, rng.random())
    return p


def _smoke_map(height, width, seed):
    """Low-frequency noise in [0, 1], [1, 1, H, W] (gaussian-blurred upsampled white noise)."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(1, 1, max(2, height // 16), max(2, width // 16), generator=g)
    noise = F.interpolate(noise, size=(height, width), mode="bicubic", align_corners=False)
    noise = TF.gaussian_blur(noise, kernel_size=31, sigma=8.0)
    lo, hi = noise.min(), noise.max()
    return (noise - lo) / (hi - lo + 1e-6)


def _specular_mask(height, width, blobs):
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32),
                            indexing="ij")
    m = torch.zeros(height, width)
    for cy, cx, ry, rx, th in blobs:
        dy, dx = yy - cy, xx - cx
        u = dx * math.cos(th) + dy * math.sin(th)
        v = -dx * math.sin(th) + dy * math.cos(th)
        d = (u / rx) ** 2 + (v / ry) ** 2
        m = torch.maximum(m, torch.clamp(1.5 - d, 0, 1))  # saturated core, soft rim
    return m


def apply(clip, p):
    """
    Apply one parameter set to a clip ``[T, C, H, W]`` float in [0, 1]; returns the same shape.
    """
    t, c, height, width = clip.shape
    x = clip
    if p.get("flip"):
        x = x.flip(-1)
    if "crop" in p:
        top, left, h, w = p["crop"]
        x = TF.resized_crop(x, top, left, h, w, [height, width], interpolation=InterpolationMode.BILINEAR,
                            antialias=True)
    if p.get("rotate"):
        x = TF.rotate(x, p["rotate"], interpolation=InterpolationMode.BILINEAR)
    if "color" in p:
        b, ct, s = p["color"]
        x = TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(x, b), ct), s)
    if "rand_aug" in p:
        from torchvision.transforms import v2

        ra = p["rand_aug"]
        state = torch.get_rng_state()
        torch.manual_seed(ra["seed"])  # v2.RandAugment draws its ops once per call: one call per clip
        u8 = (x.clamp(0, 1) * 255).round().to(torch.uint8)
        x = v2.RandAugment(num_ops=ra["num_ops"], magnitude=ra["magnitude"])(u8).float() / 255.0
        torch.set_rng_state(state)
    if "blur_sigma" in p:
        s = p["blur_sigma"]
        k = 2 * int(math.ceil(3 * s)) + 1
        x = TF.gaussian_blur(x, kernel_size=k, sigma=s)
    if "smoke" in p:
        sm = p["smoke"]
        veil = _smoke_map(height, width, sm["seed"])[0]  # [1, H, W], same for every frame
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = mean + (x - mean) * sm["contrast"]
        x = x * (1 - sm["alpha"] * veil) + sm["alpha"] * veil * sm["grey"]
    if "specular" in p:
        m = _specular_mask(height, width, p["specular"])
        x = x * (1 - m) + m
    if "erase" in p:
        top, left, h, w, v = p["erase"]
        x = x.clone()
        x[..., top : top + h, left : left + w] = v
    return x.clamp(0, 1)


def temporal_shift(max_rows, rng=random):
    """Offset (rows) of the clip centre from the keyframe row, uniform in [-max_rows, max_rows]."""
    return rng.randint(-max_rows, max_rows) if max_rows > 0 else 0
