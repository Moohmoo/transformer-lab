"""Activation components used by Transformer blocks."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class ReLUActivation(nn.Module):
    """ReLU wrapper for explicit component selection."""

    def forward(self, x: Tensor) -> Tensor:
        return functional.relu(x)


class GELUActivation(nn.Module):
    """GELU wrapper for explicit component selection."""

    def forward(self, x: Tensor) -> Tensor:
        return functional.gelu(x)


class SwiGLUActivation(nn.Module):
    """SwiGLU gating activation.

    Expects two tensors with the same shape and returns their gated product.
    """

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        return functional.silu(gate) * up


def create_pointwise_activation(name: str) -> nn.Module:
    """Builds pointwise activation by name."""
    registry: dict[str, type[nn.Module]] = {
        "relu": ReLUActivation,
        "gelu": GELUActivation,
    }
    if name not in registry:
        options = ", ".join(registry.keys())
        raise ValueError(f"Unknown activation '{name}'. Available: {options}")
    return registry[name]()
