from pathlib import Path

import torch

from src.training.checkpoint_store import CheckpointStore


def test_checkpoint_store_save_and_load(tmp_path: Path) -> None:
    device = torch.device("cpu")
    store = CheckpointStore(tmp_path, device)

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    initial_weight = model.weight.detach().clone()

    store.save(
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=42,
        best_val_loss=1.23,
        config={"name": "test"},
        is_best=True,
    )

    with torch.no_grad():
        model.weight.add_(1.0)

    checkpoint = store.load(model=model, optimizer=optimizer)

    assert torch.allclose(model.weight, initial_weight)
    assert checkpoint.epoch == 2
    assert checkpoint.global_step == 42
    assert checkpoint.best_val_loss == 1.23
    assert store.best_path.exists()
