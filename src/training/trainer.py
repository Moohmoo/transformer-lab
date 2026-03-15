"""Training pipeline and CLI entrypoint."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.loader import build_dataloaders
from src.data.tokenizer import CharTokenizer
from src.model.builder import build_transformer
from src.training.checkpoint_store import CheckpointStore
from src.training.config import TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    """Computes linear warmup + cosine decay learning rate."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Reusable training orchestration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainConfig,
        checkpoint_store: CheckpointStore,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.checkpoint_store = checkpoint_store
        self.device = device
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

    def maybe_resume(self) -> None:
        """Restores model/optimizer state from latest checkpoint when enabled."""
        # Resume is opt-in via config so interrupted long runs can continue safely.
        if self.config.resume and self.checkpoint_store.has_checkpoint():
            checkpoint = self.checkpoint_store.load(self.model, self.optimizer)
            self.start_epoch = checkpoint.epoch + 1
            self.global_step = checkpoint.global_step
            self.best_val_loss = checkpoint.best_val_loss
            print(
                f"resume_epoch={self.start_epoch + 1} global_step={self.global_step} "
                f"best_val_loss={self.best_val_loss:.4f}"
            )

    def train_one_epoch(self, train_loader: DataLoader[tuple[Tensor, Tensor]], max_steps: int, epoch: int) -> float:
        """Runs one training epoch.

        Args:
            train_loader: Loader yielding ``(input_ids, targets)`` batches.
            max_steps: Total optimization steps across training.
            epoch: Zero-based epoch index.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            lr = get_lr(self.global_step, self.config.warmup_steps, max_steps, self.config.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            logits = self.model(input_ids)
            loss = functional.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            if batch_idx % 50 == 0:
                print(
                    f"epoch={epoch + 1} batch={batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={lr:.6f}"
                )

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader[tuple[Tensor, Tensor]]) -> tuple[float, float]:
        """Evaluates model on validation data.

        Args:
            val_loader: Loader yielding ``(input_ids, targets)`` batches.

        Returns:
            Tuple of ``(mean_loss, perplexity)``.
        """
        self.model.eval()
        total_loss = 0.0

        for input_ids, targets in val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(input_ids)
            loss = functional.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        perplexity = math.exp(min(avg_loss, 20.0))
        return avg_loss, perplexity

    def fit(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        val_loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> None:
        """Runs the full train/validate loop with checkpointing."""
        self.maybe_resume()
        max_steps = len(train_loader) * self.config.num_epochs
        train_start_time = time.time()

        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            train_loss = self.train_one_epoch(train_loader=train_loader, max_steps=max_steps, epoch=epoch)
            val_loss, perplexity = self.evaluate(val_loader)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.checkpoint_store.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                global_step=self.global_step,
                best_val_loss=self.best_val_loss,
                config=asdict(self.config),
                is_best=is_best,
            )

            elapsed = time.time() - epoch_start_time
            print(
                f"epoch={epoch + 1}/{self.config.num_epochs} time={elapsed:.1f}s "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ppl={perplexity:.2f} "
                f"is_best={is_best}"
            )

        total_time = time.time() - train_start_time
        print(f"training_time={total_time:.1f}s best_val_loss={self.best_val_loss:.4f}")
        print(f"checkpoints_dir={self.config.checkpoint_dir}")


def generate_sample(
    model: nn.Module,
    tokenizer: CharTokenizer,
    device: torch.device,
    max_context: int,
    max_len: int = 200,
    prompt: str = "\n",
    temperature: float = 0.8,
) -> str:
    """Generates autoregressive text sample."""
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt, drop_unknown=True), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_len):
            context = input_ids[:, -max_context:]
            logits = model(context)
            next_logits = logits[:, -1, :] / temperature
            probs = functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    generated = tokenizer.decode(input_ids[0].tolist())
    print("generated_sample=")
    print(generated)
    return generated


def build_model_and_optimizer(config: TrainConfig, device: torch.device) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Builds model and optimizer from train config."""
    model = build_transformer(config.to_model_config()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    return model, optimizer


def train_model(config: TrainConfig) -> nn.Module:
    """Runs end-to-end training."""
    device = DEVICE
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"dataset_type={config.dataset_type}")
    print(f"attention={config.attention_type}")
    if config.token_metadata_path is not None:
        print(f"token_metadata_path={config.token_metadata_path}")
        if config.train_tokens_path is not None:
            print(f"train_tokens_path={config.train_tokens_path}")
        if config.val_tokens_path is not None:
            print(f"val_tokens_path={config.val_tokens_path}")
    elif config.train_text_path is not None and config.val_text_path is not None:
        print(f"train_text_path={config.train_text_path}")
        print(f"val_text_path={config.val_text_path}")
    elif config.text_path is not None:
        print(f"text_path={config.text_path}")

    train_loader, val_loader, tokenizer = build_dataloaders(config)
    model, optimizer = build_model_and_optimizer(config, device)

    print(f"parameters_total={model.count_parameters():,}")
    print(f"parameters_trainable={model.count_trainable_parameters():,}")

    checkpoint_store = CheckpointStore(config.checkpoint_dir, device)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        checkpoint_store=checkpoint_store,
        device=device,
    )
    trainer.fit(train_loader=train_loader, val_loader=val_loader)

    if tokenizer is not None:
        generate_sample(model=model, tokenizer=tokenizer, device=device, max_context=config.max_seq_len)

    return model


def parse_args() -> TrainConfig:
    """Parses CLI arguments into TrainConfig."""
    parser = argparse.ArgumentParser(description="Transformer Research Lab training")

    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--attention_type", type=str, default="standard", choices=["standard", "linear"])
    parser.add_argument("--norm_type", type=str, default="rmsnorm", choices=["rmsnorm", "layernorm"])
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "relu", "gelu"])
    parser.add_argument("--positional_type", type=str, default="rope", choices=["rope", "none"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--no_resume", action="store_true")

    parser.add_argument("--text_path", type=str, default="", help="Path to a plain-text corpus file.")
    parser.add_argument("--train_text_path", type=str, default="", help="Path to train corpus text file.")
    parser.add_argument("--val_text_path", type=str, default="", help="Path to validation corpus text file.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="raw_text",
        choices=["raw_text", "instruction", "domain_text"],
        help="Dataset family adapter.",
    )

    parser.add_argument("--train_instruction_path", type=str, default="")
    parser.add_argument("--val_instruction_path", type=str, default="")
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--input_field", type=str, default="input")
    parser.add_argument("--output_field", type=str, default="output")

    parser.add_argument("--train_domain_text_path", type=str, default="")
    parser.add_argument("--val_domain_text_path", type=str, default="")

    parser.add_argument(
        "--token_metadata_path",
        type=str,
        default="",
        help="Path to token metadata JSON produced by data loader.",
    )
    parser.add_argument("--train_tokens_path", type=str, default="", help="Optional override for train token file.")
    parser.add_argument("--val_tokens_path", type=str, default="", help="Optional override for val token file.")
    parser.add_argument("--train_split", type=float, default=0.9)

    args = parser.parse_args()
    return TrainConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=None,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        attention_type=args.attention_type,
        norm_type=args.norm_type,
        ffn_type=args.ffn_type,
        positional_type=args.positional_type,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seq_len=args.seq_len,
        checkpoint_dir=Path(args.checkpoint_dir),
        resume=not args.no_resume,
        text_path=Path(args.text_path) if args.text_path else None,
        train_text_path=Path(args.train_text_path) if args.train_text_path else None,
        val_text_path=Path(args.val_text_path) if args.val_text_path else None,
        token_metadata_path=Path(args.token_metadata_path) if args.token_metadata_path else None,
        train_tokens_path=Path(args.train_tokens_path) if args.train_tokens_path else None,
        val_tokens_path=Path(args.val_tokens_path) if args.val_tokens_path else None,
        train_split=args.train_split,
        dataset_type=args.dataset_type,
        train_instruction_path=Path(args.train_instruction_path) if args.train_instruction_path else None,
        val_instruction_path=Path(args.val_instruction_path) if args.val_instruction_path else None,
        instruction_field=args.instruction_field,
        input_field=args.input_field,
        output_field=args.output_field,
        train_domain_text_path=Path(args.train_domain_text_path) if args.train_domain_text_path else None,
        val_domain_text_path=Path(args.val_domain_text_path) if args.val_domain_text_path else None,
    )


def main() -> None:
    """CLI entrypoint."""
    train_model(parse_args())


if __name__ == "__main__":
    main()
