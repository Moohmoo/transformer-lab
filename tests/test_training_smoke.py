from pathlib import Path

import torch

from src.data.loader import build_dataloaders
from src.training.checkpoint_store import CheckpointStore
from src.training.config import TrainConfig
from src.training.trainer import Trainer, build_model_and_optimizer


def test_trainer_smoke_fit_writes_checkpoint(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    checkpoint_dir = tmp_path / "checkpoints"

    train_path.write_text("hello world " * 40, encoding="utf-8")
    val_path.write_text("hello world " * 20, encoding="utf-8")

    config = TrainConfig(
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_layers=1,
        max_seq_len=32,
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-3,
        warmup_steps=1,
        seq_len=8,
        checkpoint_dir=checkpoint_dir,
        resume=False,
        dataset_type="raw_text",
        train_text_path=train_path,
        val_text_path=val_path,
        attention_type="standard",
        positional_type="none",
    )

    train_loader, val_loader, tokenizer = build_dataloaders(config)
    assert tokenizer is not None
    assert config.vocab_size == tokenizer.vocab_size

    model, optimizer = build_model_and_optimizer(config, torch.device("cpu"))
    store = CheckpointStore(checkpoint_dir, torch.device("cpu"))
    trainer = Trainer(model, optimizer, config, store, torch.device("cpu"))

    trainer.fit(train_loader=train_loader, val_loader=val_loader)

    assert store.latest_path.exists()
    assert trainer.global_step > 0
