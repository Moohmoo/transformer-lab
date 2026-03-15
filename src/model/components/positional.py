"""Positional encoding components."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionalEmbedding(nn.Module):
    """Precomputes rotary frequencies for one attention head."""

    def __init__(self, d_model: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: Tensor) -> Tensor:
    d_half = x.shape[-1] // 2
    return torch.cat([-x[..., d_half:], x[..., :d_half]], dim=-1)


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Applies RoPE to queries and keys."""
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    return q * cos_b + _rotate_half(q) * sin_b, k * cos_b + _rotate_half(k) * sin_b
