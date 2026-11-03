"""
Global temporal-context modules on top of frozen per-second clip features.

Every module maps ``(B, T, D) -> (B, T, C)`` and takes a ``causal`` flag: with
``causal=True`` the output at second ``t`` depends on inputs ``<= t`` only.
``forward_all`` returns the outputs that receive a loss (one per refinement stage
for G2, a single one otherwise), the last one being the prediction.

  G0  per-second MLP, no temporal context (lower bound: the clip alone)
  G1  local TCN, one stage, dilations 1..32 (receptive field +/- 63 s)
  G2  MS-TCN, 2 stages x dilations 1..512 (`tools/train_temporal.TemporalModel`)
  G3  bidirectional GRU (unidirectional GRU when causal)
  G4  transformer on 8x average-pooled tokens with T5-style relative position bias
      (causal mask, and every second reads only completed tokens, when causal)
  G5  G1 + G4 on the TCN features, fused residually
  G6  LSTR-style: long memory of pooled tokens compressed by learned queries +
      attention over a short window of seconds (causal only)

Defaults keep every module at roughly 0.3-0.5 M parameters with D = 1536 and
C = 15 phases + 33 events; ``count_parameters`` reports the exact number.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from train_temporal import DilatedResidual, TemporalModel  # noqa: E402

CONTEXTS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6")


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class ContextModule(nn.Module):
    def forward_all(self, x):
        return [self(x)]


# ---- G0 ------------------------------------------------------------------------------
class PerSecondMLP(ContextModule):
    def __init__(self, dim_in, n_out, causal=False, hidden=192, dropout=0.5, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ---- G1 ------------------------------------------------------------------------------
class LocalTCN(nn.Module):
    """Single-stage dilated residual TCN trunk, ``(B, T, D) -> (B, T, channels)``."""

    def __init__(self, dim_in, channels, layers, causal, dropout):
        super().__init__()
        self.inp = nn.Conv1d(dim_in, channels, 1)
        self.layers = nn.ModuleList(DilatedResidual(2**i, channels, causal, dropout) for i in range(layers))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for layer in self.layers:
            h = layer(h)
        return h.transpose(1, 2)


class LocalTCNHead(ContextModule):
    def __init__(self, dim_in, n_out, causal=False, channels=96, layers=6, dropout=0.5, **_):
        super().__init__()
        self.trunk = LocalTCN(dim_in, channels, layers, causal, dropout)
        self.head = nn.Linear(channels, n_out)

    def forward(self, x):
        return self.head(self.trunk(x))


# ---- G2 ------------------------------------------------------------------------------
class MSTCN(ContextModule):
    """`train_temporal.TemporalModel` behind the common interface (same init order, same seeds)."""

    def __init__(self, dim_in, n_out, causal=False, n_phases=None, channels=64, layers=10, stages=2, dropout=0.5, **_):
        super().__init__()
        assert n_phases is not None, "MS-TCN refinement stages need the phase/event split"
        self.model = TemporalModel(dim_in, n_phases, n_out - n_phases, channels, layers, stages, causal, dropout)

    def forward_all(self, x):
        return [torch.cat([p, e], 1).transpose(1, 2) for p, e in self.model(x.transpose(1, 2))]

    def forward(self, x):
        return self.forward_all(x)[-1]


# ---- G3 ------------------------------------------------------------------------------
class RecurrentContext(ContextModule):
    def __init__(self, dim_in, n_out, causal=False, hidden=None, layers=2, dropout=0.5, **_):
        super().__init__()
        hidden = hidden or (128 if causal else 96)
        self.inp = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=not causal, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (1 if causal else 2), n_out)

    def forward(self, x):
        h, _ = self.rnn(self.inp(x))
        return self.head(self.drop(h))


# ---- attention helpers -------------------------------------------------------------------
def relative_bucket(rel, causal, num_buckets=32, max_distance=512):
    """
    T5 relative-position buckets (``rel = key - query``): exact for small offsets, log-spaced beyond.

    >>> relative_bucket(torch.tensor([-3, 0, 3]), causal=False).tolist()
    [3, 0, 19]
    """
    if causal:
        n = -torch.clamp(rel, max=0)
        offset = torch.zeros_like(rel)
    else:
        num_buckets //= 2
        offset = (rel > 0).long() * num_buckets
        n = rel.abs()
    exact = num_buckets // 2
    large = exact + (
        torch.log(n.float().clamp(min=1) / exact) / math.log(max_distance / exact) * (num_buckets - exact)
    ).long()
    large = torch.clamp(large, max=num_buckets - 1)
    return offset + torch.where(n < exact, n, large)


class RelativeSelfAttention(nn.Module):
    def __init__(self, dim, heads, causal, dropout, num_buckets=32, max_distance=512):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.num_buckets, self.max_distance = num_buckets, max_distance
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.bias = nn.Embedding(num_buckets, heads)
        self.drop = dropout

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        pos = torch.arange(N, device=x.device)
        rel = pos[None, :] - pos[:, None]
        bias = self.bias(relative_bucket(rel, self.causal, self.num_buckets, self.max_distance)).permute(2, 0, 1)
        if self.causal:
            bias = bias.masked_fill(rel[None] > 0, float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias[None].to(q.dtype), dropout_p=self.drop if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, causal, dropout, ffn=2):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = RelativeSelfAttention(dim, heads, causal, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        return x + self.drop(self.ffn(self.n2(x)))


def pool_tokens(h, stride):
    """Average ``(B, T, d)`` over blocks of ``stride`` seconds -> ``(B, ceil(T / stride), d)`` (ragged tail averaged over its own length)."""
    B, T, d = h.shape
    n = -(-T // stride)
    pad = n * stride - T
    s = F.pad(h, (0, 0, 0, pad)).view(B, n, stride, d).sum(2)
    count = torch.full((n,), float(stride), device=h.device)
    count[-1] = stride - pad
    return s / count[None, :, None]


def unpool_tokens(z, T, stride, causal):
    """
    Token outputs back to seconds. Offline: second t reads token t // stride. Causal:
    second t reads the last *completed* token, (t + 1) // stride - 1, and zeros before the first.
    """
    t = torch.arange(T, device=z.device)
    if not causal:
        return z[:, t // stride]
    z = F.pad(z, (0, 0, 1, 0))
    return z[:, (t + 1) // stride]


# ---- G4 ------------------------------------------------------------------------------
class GlobalTransformer(nn.Module):
    """``(B, T, d) -> (B, T, d)`` context read from attention over pooled tokens of the whole sequence."""

    def __init__(self, dim, layers, heads, causal, dropout, stride=8):
        super().__init__()
        self.stride, self.causal = stride, causal
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, causal, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        z = pool_tokens(h, self.stride)
        for block in self.blocks:
            z = block(z)
        return unpool_tokens(self.norm(z), h.shape[1], self.stride, self.causal)


class GlobalTransformerHead(ContextModule):
    def __init__(self, dim_in, n_out, causal=False, dim=96, layers=3, heads=4, stride=8, dropout=0.5, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(dim_in, dim), nn.Dropout(dropout))
        self.ctx = GlobalTransformer(dim, layers, heads, causal, min(dropout, 0.1), stride)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, n_out))

    def forward(self, x):
        h = self.inp(x)
        return self.head(h + self.ctx(h))


# ---- G5 ------------------------------------------------------------------------------
class LocalGlobalHead(ContextModule):
    def __init__(self, dim_in, n_out, causal=False, dim=80, tcn_layers=6, layers=2, heads=4, stride=8, dropout=0.5, **_):
        super().__init__()
        self.local = LocalTCN(dim_in, dim, tcn_layers, causal, dropout)
        self.ctx = GlobalTransformer(dim, layers, heads, causal, min(dropout, 0.1), stride)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, n_out))

    def forward(self, x):
        h = self.local(x)
        return self.head(h + self.ctx(h))


# ---- G6 ------------------------------------------------------------------------------
def _attend(q, k, v, heads, mask=None):
    """
    Batched multi-head attention: q [*lead, Nq, D], k/v [*lead, Nk, D], optional mask
    [*lead, 1 or heads, Nq, Nk] (bool: True = attend, or additive float).
    """
    *lead, nq, D = q.shape
    nk = k.shape[-2]
    split = lambda t, n: t.reshape(-1, n, heads, D // heads).transpose(1, 2)  # noqa: E731
    out = F.scaled_dot_product_attention(
        split(q, nq), split(k, nk), split(v, nk), attn_mask=None if mask is None else mask.reshape(-1, mask.shape[-3], nq, nk)
    )
    return out.transpose(1, 2).reshape(*lead, nq, D)


class LongShortMemory(ContextModule):
    """
    LSTR-style streaming context (Xu et al., NeurIPS 2021), dense over every second.

    * long memory: the last ``memory`` completed tokens (``stride`` s each, so 34 min by
      default), compressed into ``queries`` latents by cross-attention, then one
      self-attention layer among the latents;
    * short memory: the last ``window`` seconds; each second attends to the latents
      plus its causal window (learned per-offset bias), two decoder layers.
    """

    def __init__(self, dim_in, n_out, causal=True, dim=96, heads=4, stride=8, memory=256, queries=16, window=32,
                 dec_layers=2, dropout=0.5, **_):
        super().__init__()
        if not causal:
            raise ValueError("G6 (LSTR-style memory) is causal only")
        self.heads, self.stride, self.memory, self.window = heads, stride, memory, window
        self.inp = nn.Sequential(nn.Linear(dim_in, dim), nn.Dropout(dropout))
        self.mem_pos = nn.Parameter(torch.zeros(memory, dim))
        self.null = nn.Parameter(torch.zeros(1, dim))  # always-visible slot: an empty memory still has a key
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.enc_q, self.enc_kv = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim)
        self.enc_ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.latent = TransformerBlock(dim, heads, causal=False, dropout=0.1)
        self.dec = nn.ModuleList(
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim), "q": nn.Linear(dim, dim), "kv": nn.Linear(dim, 2 * dim),
                "proj": nn.Linear(dim, dim),
                "ffn": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)),
            })
            for _ in range(dec_layers)
        )
        self.win_bias = nn.Parameter(torch.zeros(dec_layers, heads, window))
        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_out))

    def compress(self, h):
        """Latents ``(B, N + 1, Q, d)``: entry k summarises the tokens completed before token k."""
        B, T, d = h.shape
        L = self.memory
        z = pool_tokens(h, self.stride)[:, : T // self.stride]  # completed tokens only
        n = z.shape[1]
        zp = F.pad(z, (0, 0, L, 0))
        win = zp.unfold(1, L, 1).permute(0, 1, 3, 2)[:, : n + 1]  # (B, n+1, L, d): window k = z[k-L : k]
        valid = (torch.arange(n + 1, device=h.device)[:, None] + torch.arange(L, device=h.device)[None]) >= L
        win = torch.cat([self.null.expand(B, n + 1, 1, d), win + self.mem_pos], 2)
        mask = torch.cat([torch.ones(n + 1, 1, dtype=torch.bool, device=h.device), valid], 1)
        k, v = self.enc_kv(win).chunk(2, -1)
        q = self.enc_q(self.queries).expand(B, n + 1, -1, -1)
        lat = q + _attend(q, k, v, self.heads, mask[None, :, None, None, :].expand(B, -1, 1, q.shape[2], -1))
        lat = lat + self.enc_ffn(lat)
        Q = lat.shape[2]
        return self.latent(lat.reshape(B * (n + 1), Q, d)).reshape(B, n + 1, Q, d)

    def forward(self, x):
        h = self.inp(x)
        B, T, d = h.shape
        S = self.window
        lat = self.compress(h)[:, (torch.arange(T, device=h.device) + 1) // self.stride]  # (B, T, Q, d)
        Q = lat.shape[2]
        valid = (torch.arange(T, device=h.device)[:, None] + torch.arange(S, device=h.device)[None]) >= S - 1
        for i, layer in enumerate(self.dec):
            hn = layer["norm"](h)
            win = F.pad(hn, (0, 0, S - 1, 0)).unfold(1, S, 1).permute(0, 1, 3, 2)  # (B, T, S, d): h[t-S+1 .. t]
            kv = torch.cat([lat, win], 2)
            k, v = layer["kv"](kv).chunk(2, -1)
            # additive mask: latents free, window masked before the video start, per-offset bias
            wb = self.win_bias[i].flip(-1)  # offset S-1 .. 0 aligned with the window order
            bias = torch.cat(
                [torch.zeros(self.heads, T, Q, device=h.device),
                 wb[:, None, :].expand(-1, T, -1).masked_fill(~valid[None], float("-inf"))], 2
            )
            q = layer["q"](hn)[:, :, None]  # (B, T, 1, d)
            att = _attend(q, k, v, self.heads, bias.permute(1, 0, 2)[None, :, :, None, :].expand(B, -1, -1, -1, -1).to(q.dtype))
            h = h + self.drop(layer["proj"](att[:, :, 0]))
            h = h + self.drop(layer["ffn"](h))
        return self.head(h)


# ---- factory ----------------------------------------------------------------------------
REGISTRY = {
    "G0": PerSecondMLP,
    "G1": LocalTCNHead,
    "G2": MSTCN,
    "G3": RecurrentContext,
    "G4": GlobalTransformerHead,
    "G5": LocalGlobalHead,
    "G6": LongShortMemory,
}


def build_context(name, dim_in, n_out, causal=False, **kwargs):
    """
    >>> m = build_context("G1", 16, 5, causal=True)
    >>> tuple(m(torch.zeros(2, 40, 16)).shape)
    (2, 40, 5)
    """
    return REGISTRY[name](dim_in, n_out, causal=causal, **kwargs)
