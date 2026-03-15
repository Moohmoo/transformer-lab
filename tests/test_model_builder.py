import pytest
import torch

from src.model.builder import build_transformer
from src.model.model_config import ModelConfig


def test_build_transformer_forward_shape() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        max_seq_len=32,
        attention_type="standard",
    )
    model = build_transformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(input_ids)

    assert logits.shape == (2, 8, config.vocab_size)
    assert model.count_parameters() >= model.count_trainable_parameters()


def test_build_transformer_rejects_invalid_config() -> None:
    config = ModelConfig(vocab_size=32, d_model=10, n_heads=3)

    with pytest.raises(ValueError, match="divisible"):
        build_transformer(config)
