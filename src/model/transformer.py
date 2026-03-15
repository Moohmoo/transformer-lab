"""Decoder-only Transformer model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.model.blocks.transformer_block import TransformerBlock
from src.model.components.norms import create_norm
from src.model.model_config import ModelConfig


class Transformer(nn.Module):
    """Decoder-only Transformer assembled from modular components."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    attention_type=config.attention_type,
                    norm_type=config.norm_type,
                    ffn_type=config.ffn_type,
                    positional_type=config.positional_type,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = create_norm(config.norm_type, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: Tensor) -> Tensor:
        _, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        mask = self._make_causal_mask(seq_len, x.device)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
