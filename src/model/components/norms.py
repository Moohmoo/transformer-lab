"""Normalization components."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + self.eps)
        return (x / rms.type_as(x)) * self.weight


def create_norm(norm_type: str, d_model: int) -> nn.Module:
    """Builds normalization module by name."""
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")
