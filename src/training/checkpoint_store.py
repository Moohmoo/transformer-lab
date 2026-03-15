"""Checkpoint persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CheckpointData:
    """Restored checkpoint metadata."""

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    epoch: int
    global_step: int
    best_val_loss: float
    config: dict[str, Any]


class CheckpointStore:
    """Manages training checkpoints."""

    latest_filename = "checkpoint_latest.pth"
    best_filename = "best_model.pth"

    def __init__(self, checkpoint_dir: str | Path, device: torch.device) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_dir / self.latest_filename

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / self.best_filename

    def has_checkpoint(self) -> bool:
        return self.latest_path.exists()

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        best_val_loss: float,
        config: dict[str, Any],
        is_best: bool = False,
    ) -> None:
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        torch.save(payload, self.latest_path)
        print(f"checkpoint_saved={self.latest_path}")

        if is_best:
            torch.save(payload, self.best_path)
            print(f"best_checkpoint_saved={self.best_path}")

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> CheckpointData:
        if not self.has_checkpoint():
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

        payload = torch.load(self.latest_path, map_location=self.device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        print(f"checkpoint_loaded={self.latest_path}")
        return CheckpointData(
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload.get("optimizer_state_dict", {}),
            epoch=payload.get("epoch", 0),
            global_step=payload.get("global_step", 0),
            best_val_loss=payload.get("best_val_loss", float("inf")),
            config=payload.get("config", {}),
        )

    def load_best(self, model: nn.Module) -> None:
        if not self.best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {self.best_path}")

        payload = torch.load(self.best_path, map_location=self.device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        print(f"best_checkpoint_loaded={self.best_path}")
