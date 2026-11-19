# Pure-PyTorch Mamba2 (SSD) block and a bidirectional encoder for per-second sequences.
#
# The block and the chunked SSD scan are adapted from mamba2-minimal
# (https://github.com/tommyip/mamba2-minimal), Copyright 2024 Tommy Ip,
# licensed under the Apache License, Version 2.0, which is itself a port of the
# reference SSD of state-spaces/mamba (Copyright 2024 Tri Dao, Albert Gu, Apache-2.0).
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
# Changes from mamba2-minimal: no einops, the four-operand einsum of the diagonal blocks is
# split into two contractions (bounded memory on hour-long videos), sequences are padded to a
# multiple of the chunk size internally, no inference cache, one B/C group, and a
# bidirectional residual encoder (`BiMamba2Encoder`) on top. No custom CUDA kernels: runs on CPU.
# Copied from branch exp7-phase-model; EXP-21 adds ``bidirectional=False`` (forward scan only: causal encoder).
"""
Pure-PyTorch Mamba2 for the phase models of Experiment 7.

>>> torch.manual_seed(0) and None
>>> blk = Mamba2(d_model=8, d_state=4, headdim=4, chunk_size=4)
>>> x = torch.randn(2, 10, 8)
>>> y = blk(x)
>>> y.shape
torch.Size([2, 10, 8])
>>> torch.allclose(y, _reference_scan(blk, x), atol=1e-4)  # chunked SSD == naive recurrence
True
>>> y2 = blk(torch.cat([x, torch.randn(2, 5, 8)], 1))[:, :10]  # causal: the future does not change the past
>>> torch.allclose(y, y2, atol=1e-5)
True
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def segsum(x):
    """Stable segment sum: ``out[..., i, j] = sum(x[..., j+1 : i+1])`` for ``j <= i``, ``-inf`` above."""
    t = x.size(-1)
    x = x[..., None].expand(*x.shape, t)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd(x, a, b, c, chunk_size):
    """
    Chunked state-space-duality scan (Mamba2, Listing 1 of Dao & Gu 2024).

    x: (batch, seqlen, heads, headdim), a: (batch, seqlen, heads),
    b, c: (batch, seqlen, heads, d_state); seqlen must be a multiple of ``chunk_size``.
    Returns y: (batch, seqlen, heads, headdim).
    """
    bs, sl, h, p = x.shape
    n = b.shape[-1]
    nc, l = sl // chunk_size, chunk_size
    x = x.reshape(bs, nc, l, h, p)
    b = b.reshape(bs, nc, l, h, n)
    c = c.reshape(bs, nc, l, h, n)
    a = a.reshape(bs, nc, l, h).permute(0, 3, 1, 2)  # b h c l
    a_cumsum = torch.cumsum(a, dim=-1)

    # 1. intra-chunk (diagonal blocks): y = (C B^T * L) x, in two contractions
    lmat = torch.exp(segsum(a))  # b h c l s
    cb = torch.einsum("bclhn,bcshn->bhcls", c, b) * lmat
    y_diag = torch.einsum("bhcls,bcshp->bclhp", cb, x)

    # 2. state at the end of every chunk
    decay_states = torch.exp(a_cumsum[:, :, :, -1:] - a_cumsum)  # b h c l
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, x)

    # 3. inter-chunk recurrence
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(a_cumsum[:, :, :, -1], (1, 0))))  # b h (c+1) (c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]

    # 4. state -> output
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states, torch.exp(a_cumsum))
    return (y_diag + y_off).reshape(bs, sl, h, p)


class RMSNormGated(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z=None):
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2(nn.Module):
    """Causal Mamba2 block ``(B, T, d_model) -> (B, T, d_model)`` (one B/C group)."""

    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, headdim=16, chunk_size=64,
                 dt_min=0.001, dt_max=0.1, a_init=(1.0, 16.0)):
        super().__init__()
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0
        self.d_state, self.headdim, self.chunk_size = d_state, headdim, chunk_size
        self.nheads = self.d_inner // headdim
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(*a_init)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _inputs(self, u):
        """Projections shared by `forward` and the naive reference scan."""
        t = u.shape[1]
        z, xbc, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], -1)
        dt = F.softplus(dt + self.dt_bias)  # b t h
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[:, :, :t].transpose(1, 2))
        x, b, c = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], -1)
        return z, x.reshape(*x.shape[:2], self.nheads, self.headdim), b, c, dt, -torch.exp(self.A_log)

    def _output(self, y, x, z):
        y = y + x * self.D[:, None]
        y = y.reshape(*y.shape[:2], self.d_inner)
        return self.out_proj(self.norm(y, z))

    def forward(self, u):
        t = u.shape[1]
        z, x, b, c, dt, a = self._inputs(u)
        pad = (-t) % self.chunk_size
        xs, adt = x * dt[..., None], a * dt
        bh = b[:, :, None].expand(-1, -1, self.nheads, -1)
        ch = c[:, :, None].expand(-1, -1, self.nheads, -1)
        if pad:  # zero input and zero decay after the end: earlier outputs are unchanged
            xs, adt, bh, ch = (F.pad(m, (0, 0) * (m.dim() - 2) + (0, pad)) for m in (xs, adt, bh, ch))
        y = ssd(xs, adt, bh, ch, self.chunk_size)[:, :t]
        return self._output(y, x, z)


def _reference_scan(blk, u):
    """Naive sequential recurrence of `Mamba2.forward` (doctest only)."""
    z, x, b, c, dt, a = blk._inputs(u)
    bs, t, h, p = x.shape
    state = torch.zeros(bs, h, p, b.shape[-1])
    ys = []
    for i in range(t):
        da = torch.exp(dt[:, i] * a)  # b h
        state = state * da[:, :, None, None] + torch.einsum("bhp,bn->bhpn", x[:, i] * dt[:, i, :, None], b[:, i])
        ys.append(torch.einsum("bhpn,bn->bhp", state, c[:, i]))
    return blk._output(torch.stack(ys, 1), x, z)


class BiMamba2Layer(nn.Module):
    """Pre-norm residual layer: x + drop(fwd(norm x) + flip(bwd(flip(norm x)))), separate parameters."""

    def __init__(self, d_model, dropout=0.2, bidirectional=True, **kw):
        super().__init__()
        self.norm = RMSNormGated(d_model)
        self.fwd = Mamba2(d_model, **kw)
        self.bwd = Mamba2(d_model, **kw) if bidirectional else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # B T D
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = y + self.bwd(h.flip(1)).flip(1)
        return x + self.drop(y)


class BiMamba2Encoder(nn.Module):
    """
    ``(B, D_in, T) -> (B, n_out, T)``: 1x1 input projection, ``layers`` bidirectional Mamba2 layers,
    final norm, 1x1 output head (the calling convention of the MS-TCN stages).

    >>> enc = BiMamba2Encoder(12, 5, d_model=8, layers=2, d_state=4, headdim=4, chunk_size=8)
    >>> enc(torch.randn(1, 12, 30)).shape
    torch.Size([1, 5, 30])
    >>> enc = BiMamba2Encoder(12, 5, d_model=8, layers=2, d_state=4, headdim=4, chunk_size=8, bidirectional=False).eval()
    >>> x = torch.randn(1, 12, 30)
    >>> torch.allclose(enc(x)[..., :20], enc(torch.cat([x[..., :20], torch.randn(1, 12, 10)], -1))[..., :20], atol=1e-5)
    True
    """

    def __init__(self, dim_in, n_out, d_model=64, layers=4, dropout=0.2, bidirectional=True, **kw):
        super().__init__()
        self.inp = nn.Conv1d(dim_in, d_model, 1)
        self.layers = nn.ModuleList(BiMamba2Layer(d_model, dropout, bidirectional, **kw) for _ in range(layers))
        self.norm = RMSNormGated(d_model)
        self.out = nn.Conv1d(d_model, n_out, 1)

    def forward(self, x):
        h = self.inp(x).transpose(1, 2)
        for layer in self.layers:
            h = layer(h)
        return self.out(self.norm(h).transpose(1, 2))
