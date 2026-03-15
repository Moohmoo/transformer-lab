"""Feed-forward block with configurable activation."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.model.components.activations import SwiGLUActivation, create_pointwise_activation


class FFNBlock(nn.Module):
    """Configurable feed-forward block.

    Supported ``ffn_type`` values:
    - ``swiglu``
    - ``relu``
    - ``gelu``
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        dropout: float,
        ffn_type: str,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.ffn_type = ffn_type
        self.dropout = nn.Dropout(dropout)

        if ffn_type == "swiglu":
            d_ff_glu = int(2 * d_ff / 3)
            d_ff_glu = ((d_ff_glu + 7) // 8) * 8
            self.w_gate = nn.Linear(d_model, d_ff_glu, bias=False)
            self.w_up = nn.Linear(d_model, d_ff_glu, bias=False)
            self.w_down = nn.Linear(d_ff_glu, d_model, bias=False)
            self.activation = SwiGLUActivation()
        elif ffn_type in {"relu", "gelu"}:
            self.w_in = nn.Linear(d_model, d_ff, bias=False)
            self.w_out = nn.Linear(d_ff, d_model, bias=False)
            self.activation = create_pointwise_activation(ffn_type)
        else:
            raise ValueError("ffn_type must be 'swiglu', 'relu' or 'gelu'")

    def forward(self, x: Tensor) -> Tensor:
        if self.ffn_type == "swiglu":
            hidden = self.activation(self.w_gate(x), self.w_up(x))
            hidden = self.dropout(hidden)
            return self.w_down(hidden)

        hidden = self.activation(self.w_in(x))
        hidden = self.dropout(hidden)
        return self.w_out(hidden)
