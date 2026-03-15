"""Training configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.model.model_config import ModelConfig


@dataclass
class TrainConfig:
    """Configuration for data loading, optimization, and runtime.

    Dataset families:
        - ``raw_text``: generic plain text LM corpus.
        - ``instruction``: instruction/input/output JSON or JSONL.
        - ``domain_text``: domain-specific plain text corpus.
    """

    vocab_size: int = 1024
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int | None = None
    dropout: float = 0.1
    max_seq_len: int = 128
    attention_type: str = "standard"
    norm_type: str = "rmsnorm"
    ffn_type: str = "swiglu"
    positional_type: str = "rope"

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    seq_len: int = 64
    checkpoint_dir: Path = Path("./checkpoints")
    resume: bool = True

    text_path: Path | None = None
    train_text_path: Path | None = None
    val_text_path: Path | None = None
    token_metadata_path: Path | None = None
    train_tokens_path: Path | None = None
    val_tokens_path: Path | None = None
    train_split: float = 0.9

    dataset_type: str = "raw_text"

    train_instruction_path: Path | None = None
    val_instruction_path: Path | None = None
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"

    train_domain_text_path: Path | None = None
    val_domain_text_path: Path | None = None

    def to_model_config(self) -> ModelConfig:
        """Builds model configuration from training configuration."""
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            attention_type=self.attention_type,
            norm_type=self.norm_type,
            ffn_type=self.ffn_type,
            positional_type=self.positional_type,
        )
