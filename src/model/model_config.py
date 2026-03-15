"""Model configuration for TransformerLab assembly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Defines Transformer architecture and interchangeable components."""

    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int | None = None
    dropout: float = 0.1
    max_seq_len: int = 128
    attention_type: str = "linear"
    norm_type: str = "rmsnorm"
    ffn_type: str = "swiglu"
    positional_type: str = "rope"

    def validate(self) -> None:
        """Validates configuration consistency."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.attention_type not in {"standard", "linear"}:
            raise ValueError("attention_type must be 'standard' or 'linear'")
        if self.norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")
        if self.ffn_type not in {"swiglu", "relu", "gelu"}:
            raise ValueError("ffn_type must be 'swiglu', 'relu' or 'gelu'")
        if self.positional_type not in {"rope", "none"}:
            raise ValueError("positional_type must be 'rope' or 'none'")
