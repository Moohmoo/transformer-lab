"""Transformer block with configurable components."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.model.blocks.ffn_block import FFNBlock
from src.model.components.attention import BaseAttention, create_attention
from src.model.components.norms import create_norm


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None,
        dropout: float,
        attention_type: str,
        norm_type: str,
        ffn_type: str,
        positional_type: str,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.norm_attn = create_norm(norm_type, d_model)
        self.attention: BaseAttention = create_attention(
            attention_type=attention_type,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            positional_type=positional_type,
        )
        self.dropout_attn = nn.Dropout(dropout)

        self.norm_ffn = create_norm(norm_type, d_model)
        self.ffn = FFNBlock(d_model=d_model, d_ff=d_ff, dropout=dropout, ffn_type=ffn_type)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        attn_input = self.norm_attn(x)
        x = x + self.dropout_attn(self.attention(attn_input, mask))

        ffn_input = self.norm_ffn(x)
        x = x + self.dropout_ffn(self.ffn(ffn_input))
        return x
