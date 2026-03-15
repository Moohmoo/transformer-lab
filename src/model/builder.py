"""Model builder for assembling TransformerLab models."""

from __future__ import annotations

from src.model.model_config import ModelConfig
from src.model.transformer import Transformer


def build_transformer(config: ModelConfig) -> Transformer:
    """Builds and validates a Transformer model from config."""
    config.validate()
    return Transformer(config)
