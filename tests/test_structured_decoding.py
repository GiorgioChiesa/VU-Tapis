import numpy as np

from tapis.evaluate import structured_decoding as sd


def _priors():
    # 4 videos; class 1 at 10 s in phase 0, class 2 at 20 s in phase 1,
    # class 3 sometimes before and sometimes after class 2 (ambiguous order),
    # class 4 repeated many times
    events = {}
    for i, v in enumerate(["a", "b", "c", "d"]):
        t3 = 15 if i < 2 else 25
        events[v] = [(1, 10, 0), (2, 20, 1), (3, t3, 1)] + [(4, 30 + k, 1) for k in range(5 + i)]
    lengths = {v: 100 for v in events}
    return sd.estimate_priors(events, lengths, num_event_classes=5, num_phase_classes=2)


def test_priors_caps_chain_and_phase_matrix():
    p = _priors()
    assert p["caps"] == {1: 1, 2: 1, 3: 1}  # class 4 is repeated -> no cap
    assert p["chain"] == [1, 2]  # class 3 dropped: 2/3 order agrees only in half of the videos
    assert np.allclose(p["phase_given_event"][1], [1.0, 0.2])
    assert p["precedence"][1, 2] == 1.0 and p["precedence"][2, 3] == 0.5


def test_chain_decode_enforces_order_and_alpha():
    dets = [(2, 5, 0.9), (1, 10, 0.8), (2, 40, 0.6), (4, 1, 0.5)]
    out = sd.chain_decode(dets, chain=[1, 2], alpha=0.1)
    # class 2 at t=5 precedes class 1 -> rejected; class 4 is not in the chain
    assert [round(s, 3) for _, _, s in out] == [0.09, 0.8, 0.6, 0.5]


def test_phase_gate_and_smoothing_keep_shapes():
    ev = np.full((10, 3), 1 / 3)
    ph = np.tile([0.0, 1.0], (10, 1))
    m = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    gated = sd.phase_gate(ev, ph, m)
    assert gated.shape == (10, 3) and np.allclose(gated[:, 1], 0) and np.allclose(gated[:, 2], 1 / 3)
    s = sd.smooth(ph, 5)
    assert s.shape == ph.shape and np.allclose(s.sum(axis=1), 1)
