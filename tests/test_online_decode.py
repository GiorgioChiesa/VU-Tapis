"""Online decoder and streaming MS-TCN of EXP-19: batch == stream, and no access to the future."""

import doctest
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from tapis.evaluate import online_decode as od  # noqa: E402


def _probs(T=400, C=6, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 0.3, (T, C))
    for c in range(1, C):  # a few bumps per class, some close to each other
        for t0 in rng.integers(0, T, 6):
            p[max(0, t0 - 3) : t0 + 4, c] += rng.uniform(0.3, 0.7) * np.exp(-0.5 * (np.arange(max(0, t0 - 3), min(T, t0 + 4)) - t0) ** 2)
    p[-2:, 3] = 0.99  # a peak in the last seconds (end-of-video flush)
    return p


PARAMS = [
    dict(theta=0.5, window=5, delay=0),
    dict(theta=0.4, window=5, delay=5, caps={1: 1, 2: 2}, chain=[1, 2, 4], chain_rule="order"),
    dict(theta={1: 0.3, 2: 0.6}, window={1: 3, 3: 10}, delay={1: 0, 2: 2, 3: 10, 4: 5}, caps={3: 1},
         chain=[2, 4, 1], chain_rule="predecessor"),
]


def test_doctests():
    assert doctest.testmod(od).failed == 0


def test_batch_equals_stream():
    for seed in range(3):
        p = _probs(seed=seed)
        for kw in PARAMS:
            batch = od.decode_online(p, **kw)
            dec = od.StreamDecoder(p.shape[1], **kw)
            stream = []
            for row in p:
                stream += dec.step(row)
            stream += dec.flush()
            assert len(batch) == len(stream) and all(
                a[:2] == b[:2] and a[3] == b[3] and abs(a[2] - b[2]) < 1e-12 for a, b in zip(batch, stream)
            ), (seed, kw)


def test_emissions_never_use_the_future():
    p = _probs(seed=4)
    rng = np.random.default_rng(1)
    for kw in PARAMS:
        full = od.decode_online(p, **kw)
        for tau in (50, 137, 260):
            q = p.copy()
            q[tau + 1 :] = rng.uniform(0, 1, q[tau + 1 :].shape)  # rewrite everything after tau
            cut = od.decode_online(q, **kw)
            before = lambda em: [e for e in em if e[3] <= tau and e[3] < len(p) - 1]  # noqa: E731
            assert before(full) == before(cut), (kw, tau)


def test_delay_and_window_semantics():
    s = np.zeros((30, 2))
    s[10, 1], s[12, 1], s[20, 1] = 0.9, 0.95, 0.5
    # w = 5, L = 0: 10 is emitted at once, 12 is higher and also emitted (online cannot take back 10)
    assert [(t, te) for _, t, _, te in od.decode_online(s, window=5, delay=0)] == [(10, 10), (12, 12), (20, 20)]
    # L = 2: 10 is no longer a peak (12 is higher inside [10, 12]); 12 is emitted at 14
    assert [(t, te) for _, t, _, te in od.decode_online(s, window=5, delay=2)] == [(12, 14), (20, 22)]


def test_cap_and_chain():
    s = np.zeros((100, 4))
    s[10, 1], s[30, 1], s[50, 2], s[40, 3] = 0.9, 0.8, 0.7, 0.6
    em = od.decode_online(s, theta=0.5, window=5, delay=0, caps={1: 1}, alpha=0.1)
    assert [round(x[2], 3) for x in em] == [0.9, 0.08, 0.6, 0.7]
    # chain 3 -> 2 -> 1 ("order"): class 1 at 10 comes first, so 3 and 2 arrive after a later class
    em = od.decode_online(s, theta=0.5, window=5, delay=0, chain=[3, 2, 1], alpha=0.1)
    assert [round(x[2], 3) for x in em] == [0.9, 0.8, 0.06, 0.07]


def _causal_model(seed=0, dim=12):
    from train_temporal import TemporalModel

    torch.manual_seed(seed)
    m = TemporalModel(dim, 4, 5, channels=8, layers=4, stages=2, causal=True, dropout=0.0).double().eval()
    return m


def test_streaming_model_equals_full_forward_and_is_causal():
    from stream_infer import StreamingTemporalModel

    m = _causal_model()
    x = np.random.default_rng(0).normal(size=(60, 12))
    with torch.no_grad():
        ph, ev = m(torch.from_numpy(x).T[None])[-1]
    stream = StreamingTemporalModel(m)
    rows = [stream.step(x[t]) for t in range(len(x))]
    assert np.abs(np.stack([r[1] for r in rows]) - ev[0].T.numpy()).max() < 1e-9
    assert np.abs(np.stack([r[0] for r in rows]) - ph[0].T.numpy()).max() < 1e-9
    # changing the features after t does not change the outputs up to t
    y = x.copy()
    y[31:] += 5.0
    with torch.no_grad():
        _, ev2 = m(torch.from_numpy(y).T[None])[-1]
    assert torch.allclose(ev[..., :31], ev2[..., :31]) and not torch.allclose(ev[..., 31:], ev2[..., 31:])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
