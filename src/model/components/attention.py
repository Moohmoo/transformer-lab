"""Attention components and factory."""

from __future__ import annotations

import abc
import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from src.model.components.positional import RotaryPositionalEmbedding, apply_rotary_emb


class BaseAttention(nn.Module, abc.ABC):
    """Abstract multi-head attention interface."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        positional_type: str = "rope",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        if positional_type == "rope":
            self.rope: RotaryPositionalEmbedding | None = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        elif positional_type == "none":
            self.rope = None
        else:
            raise ValueError("positional_type must be 'rope' or 'none'")

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    @abc.abstractmethod
    def _attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        """Computes attention output in head space."""

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        seq_len = x.shape[1]
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        if self.rope is not None:
            cos, sin = self.rope(seq_len)
            q, k = apply_rotary_emb(q, k, cos, sin)

        attended = self._attend(q, k, v, mask)
        return self.w_o(self._merge_heads(attended))


class StandardAttention(BaseAttention):
    """Scaled dot-product attention with quadratic complexity."""

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = functional.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        return torch.matmul(weights, v)


class LinearAttention(BaseAttention):
    """Kernelized linear attention."""

    @staticmethod
    def _feature_map(x: Tensor) -> Tensor:
        return functional.elu(x) + 1.0

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        q_phi = self._feature_map(q)
        k_phi = self._feature_map(k)

        if mask is not None:
            return self._causal_linear_attention(q_phi, k_phi, v)

        kv = torch.matmul(k_phi.transpose(-2, -1), v)
        z = torch.matmul(q_phi, k_phi.sum(dim=-2, keepdim=True).transpose(-2, -1)).clamp(min=1e-6)
        return torch.matmul(q_phi, kv) / z

    def _causal_linear_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        kv = torch.einsum("bhsd,bhsv->bhsdv", k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)
        k_cumsum = torch.cumsum(k, dim=2)

        numerator = torch.einsum("bhsd,bhsdv->bhsv", q, kv_cumsum)
        denominator = torch.einsum("bhsd,bhsd->bhs", q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
        return numerator / denominator


def create_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    dropout: float,
    max_seq_len: int,
    positional_type: str,
) -> BaseAttention:
    """Builds attention module by name."""
    registry: dict[str, type[BaseAttention]] = {
        "standard": StandardAttention,
        "linear": LinearAttention,
    }
    if attention_type not in registry:
        options = ", ".join(registry.keys())
        raise ValueError(f"Unknown attention_type '{attention_type}'. Available: {options}")

    cls = registry[attention_type]
    return cls(
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        positional_type=positional_type,
    )
