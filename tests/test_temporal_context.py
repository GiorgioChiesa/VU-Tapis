import doctest

import torch

from tapis.models import temporal_context as tc

D, C, NP = 1536, 48, 15  # GraSP-LONG features, 15 phases + 33 events
OFFLINE = ("G0", "G1", "G2", "G3", "G4", "G5")
CAUSAL = ("G0", "G1", "G2", "G3", "G4", "G5", "G6")


def _model(name, causal, dim_in=D):
    torch.manual_seed(0)
    return tc.build_context(name, dim_in, C, causal=causal, n_phases=NP).eval()


def test_doctests():
    assert doctest.testmod(tc).failed == 0


def test_shapes_offline_and_causal():
    x = torch.randn(2, 203, 32)  # length not a multiple of the token stride
    for causal, names in ((False, OFFLINE), (True, CAUSAL)):
        for name in names:
            m = _model(name, causal, dim_in=32)
            outs = m.forward_all(x)
            assert all(tuple(o.shape) == (2, 203, C) for o in outs), (name, causal)
            assert torch.equal(outs[-1], m(x)), name
            assert torch.isfinite(outs[-1]).all(), (name, causal)


def test_g2_has_two_supervised_stages():
    assert len(_model("G2", False, dim_in=32).forward_all(torch.randn(1, 50, 32))) == 2


def test_g6_is_causal_only():
    try:
        tc.build_context("G6", 32, C, causal=False)
    except ValueError:
        return
    raise AssertionError("G6 accepted causal=False")


def test_parameter_budget():
    for causal, names in ((False, OFFLINE), (True, CAUSAL)):
        for name in names:
            n = tc.count_parameters(_model(name, causal))
            assert 0.2e6 < n < 0.55e6, (name, causal, n)


def test_causality():
    """Perturbing the input after t0 never changes the output up to t0."""
    T = 300
    for name in CAUSAL:
        m = _model(name, True, dim_in=32)
        x = torch.randn(1, T, 32)
        with torch.no_grad():
            ref = m(x)
            for t0 in (0, 7, 8, 100, 263):
                y = x.clone()
                y[:, t0 + 1 :] = torch.randn(1, T - t0 - 1, 32) * 5
                out = m(y)
                assert torch.allclose(out[:, : t0 + 1], ref[:, : t0 + 1], atol=1e-5), (name, t0)
                assert not torch.allclose(out[:, t0 + 1 :], ref[:, t0 + 1 :]), (name, t0)


def test_offline_models_see_the_future():
    """Sanity check of the causality test: offline context models do depend on later inputs."""
    for name in ("G1", "G2", "G3", "G4", "G5"):
        m = _model(name, False, dim_in=32)
        x = torch.randn(1, 300, 32)
        y = x.clone()
        y[:, 151:] += 5
        with torch.no_grad():
            assert not torch.allclose(m(x)[:, 150], m(y)[:, 150]), name


def test_pool_unpool():
    h = torch.arange(10.0).view(1, 10, 1)
    z = tc.pool_tokens(h, 4)
    assert z.flatten().tolist() == [1.5, 5.5, 8.5]
    assert tc.unpool_tokens(z, 10, 4, causal=False).flatten().tolist() == [1.5] * 4 + [5.5] * 4 + [8.5] * 2
    # causal: token 0 (seconds 0..3) becomes readable at second 3
    assert tc.unpool_tokens(z, 10, 4, causal=True).flatten().tolist() == [0, 0, 0, 1.5, 1.5, 1.5, 1.5, 5.5, 5.5, 5.5]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
