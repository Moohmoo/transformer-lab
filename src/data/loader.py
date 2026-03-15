"""Data loading and preparation utilities.

This module keeps all dataset responsibilities in one place:
- read raw corpora
- adapt instruction datasets
- prepare compact token files
- build PyTorch datasets/dataloaders
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import CharTokenizer, load_tokenizer

INSTRUCTION_SEPARATOR = "\n<|endofsample|>\n\n"


def load_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Loads a plain-text file.

    Args:
        path: File path.
        encoding: Text encoding.

    Returns:
        Entire file content.
    """
    return Path(path).read_text(encoding=encoding)


def load_json_records(path: str | Path, encoding: str = "utf-8") -> list[dict[str, Any]]:
    """Loads JSON or JSONL records from disk.

    Args:
        path: JSON/JSONL file path.
        encoding: Text encoding.

    Returns:
        List of dictionary records.
    """
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in file_path.read_text(encoding=encoding).splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Each JSONL line must be an object")
            records.append(payload)
        return records

    payload = json.loads(file_path.read_text(encoding=encoding))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("JSON file must contain a list of objects")
    return payload


@dataclass
class TokenizedDatasetMetadata:
    """Metadata for tokenized train/validation corpora."""

    vocab_size: int
    token_dtype: str
    tokenizer_path: str
    train_tokens_path: str
    val_tokens_path: str
    train_num_tokens: int
    val_num_tokens: int
    dataset_type: str = "raw_text"

    def save(self, path: str | Path) -> Path:
        """Saves metadata JSON to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "TokenizedDatasetMetadata":
        """Loads metadata JSON from disk."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.setdefault("dataset_type", "raw_text")
        return cls(**payload)


def token_dtype_for_vocab_size(vocab_size: int) -> np.dtype:
    """Chooses the smallest unsigned dtype that fits vocabulary size."""
    if vocab_size <= np.iinfo(np.uint8).max + 1:
        return np.dtype(np.uint8)
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def write_token_file(path: str | Path, token_ids: list[int], token_dtype: np.dtype) -> int:
    """Writes token ids to a compact binary file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(token_ids, dtype=token_dtype).tofile(output_path)
    return len(token_ids)


class TextLMDataset(Dataset[tuple[Tensor, Tensor]]):
    """Language-modeling dataset using fixed non-overlapping windows."""

    def __init__(self, data: Tensor, seq_len: int) -> None:
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (self.data.numel() - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = self.data[start:end]
        targets = self.data[start + 1 : end + 1]
        return input_ids, targets


class MemmapTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Language-modeling dataset backed by a binary token file."""

    def __init__(self, path: str | Path, seq_len: int, token_dtype: str) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.data = np.memmap(Path(path), dtype=np.dtype(token_dtype), mode="r")
        self.num_tokens = int(self.data.shape[0])

        min_tokens = seq_len + 1
        if self.num_tokens < min_tokens:
            raise ValueError(
                f"Token file too small for seq_len={seq_len}. Need at least {min_tokens} tokens, got {self.num_tokens}."
            )

    def __len__(self) -> int:
        return (self.num_tokens - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        window = np.asarray(self.data[start:end], dtype=np.int64)
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


def format_instruction_record(
    record: dict[str, Any],
    instruction_field: str,
    input_field: str,
    output_field: str,
) -> str:
    """Formats one instruction example into a single training text sample."""
    instruction = str(record.get(instruction_field, "")).strip()
    input_text = str(record.get(input_field, "")).strip()
    output = str(record.get(output_field, "")).strip()

    if not instruction or not output:
        return ""

    parts = [f"### Instruction:\n{instruction}\n"]
    if input_text:
        parts.append(f"### Input:\n{input_text}\n")
    parts.append(f"### Response:\n{output}\n")
    return "".join(parts)


def build_instruction_text(
    records: list[dict[str, Any]],
    instruction_field: str,
    input_field: str,
    output_field: str,
) -> str:
    """Converts instruction records to one LM corpus string.

    We keep a stable template because consistency usually improves SFT quality.
    """
    samples: list[str] = []
    for record in records:
        sample = format_instruction_record(record, instruction_field, input_field, output_field)
        if sample:
            samples.append(sample)
    return INSTRUCTION_SEPARATOR.join(samples)


def split_text(text: str, train_split: float) -> tuple[str, str]:
    """Splits one text corpus into train and validation segments."""
    split_idx = int(len(text) * train_split)
    return text[:split_idx], text[split_idx:]


def resolve_train_val_text(config: object) -> tuple[str, str]:
    """Builds train/validation text based on dataset type and paths.

    Supported dataset types:
    - ``raw_text``: plain text corpora.
    - ``instruction``: JSON/JSONL instruction records.
    - ``domain_text``: plain text from a specific domain corpus.
    """
    dataset_type = getattr(config, "dataset_type", "raw_text")

    if dataset_type == "raw_text":
        if getattr(config, "train_text_path", None) and getattr(config, "val_text_path", None):
            return load_text_file(config.train_text_path), load_text_file(config.val_text_path)
        if getattr(config, "text_path", None):
            return split_text(load_text_file(config.text_path), getattr(config, "train_split", 0.9))
        raise ValueError("raw_text requires text_path or train_text_path + val_text_path")

    if dataset_type == "instruction":
        train_path = getattr(config, "train_instruction_path", None)
        val_path = getattr(config, "val_instruction_path", None)
        if train_path is None or val_path is None:
            raise ValueError("instruction requires train_instruction_path + val_instruction_path")

        train_records = load_json_records(train_path)
        val_records = load_json_records(val_path)
        train_text = build_instruction_text(
            records=train_records,
            instruction_field=getattr(config, "instruction_field", "instruction"),
            input_field=getattr(config, "input_field", "input"),
            output_field=getattr(config, "output_field", "output"),
        )
        val_text = build_instruction_text(
            records=val_records,
            instruction_field=getattr(config, "instruction_field", "instruction"),
            input_field=getattr(config, "input_field", "input"),
            output_field=getattr(config, "output_field", "output"),
        )
        return train_text, val_text

    if dataset_type == "domain_text":
        train_path = getattr(config, "train_domain_text_path", None)
        val_path = getattr(config, "val_domain_text_path", None)
        if train_path and val_path:
            return load_text_file(train_path), load_text_file(val_path)
        if getattr(config, "text_path", None):
            return split_text(load_text_file(config.text_path), getattr(config, "train_split", 0.9))
        raise ValueError("domain_text requires train_domain_text_path + val_domain_text_path, or text_path")

    raise ValueError("dataset_type must be 'raw_text', 'instruction', or 'domain_text'")


def build_char_datasets(
    text: str,
    seq_len: int,
    train_split: float = 0.9,
) -> tuple[CharTokenizer, TextLMDataset, TextLMDataset]:
    """Builds character-level train and validation datasets from one text corpus."""
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    min_tokens_per_split = seq_len + 1
    if data.numel() < 2 * min_tokens_per_split:
        raise ValueError(
            "Corpus too small for requested seq_len. "
            f"Need at least {2 * min_tokens_per_split} tokens, got {data.numel()}."
        )

    split_idx = int(data.numel() * train_split)
    split_idx = max(split_idx, min_tokens_per_split)
    split_idx = min(split_idx, data.numel() - min_tokens_per_split)

    train_dataset = TextLMDataset(data[:split_idx], seq_len)
    val_dataset = TextLMDataset(data[split_idx:], seq_len)
    return tokenizer, train_dataset, val_dataset


def prepare_token_files(
    output_dir: str | Path,
    dataset_type: str = "raw_text",
    text_path: str | Path | None = None,
    train_text_path: str | Path | None = None,
    val_text_path: str | Path | None = None,
    train_split: float = 0.9,
    train_instruction_path: str | Path | None = None,
    val_instruction_path: str | Path | None = None,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    train_domain_text_path: str | Path | None = None,
    val_domain_text_path: str | Path | None = None,
) -> TokenizedDatasetMetadata:
    """Prepares train.bin/val.bin/meta.json for one of the supported dataset types."""
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.dataset_type = dataset_type
    cfg.text_path = Path(text_path) if text_path else None
    cfg.train_text_path = Path(train_text_path) if train_text_path else None
    cfg.val_text_path = Path(val_text_path) if val_text_path else None
    cfg.train_split = train_split
    cfg.train_instruction_path = Path(train_instruction_path) if train_instruction_path else None
    cfg.val_instruction_path = Path(val_instruction_path) if val_instruction_path else None
    cfg.instruction_field = instruction_field
    cfg.input_field = input_field
    cfg.output_field = output_field
    cfg.train_domain_text_path = Path(train_domain_text_path) if train_domain_text_path else None
    cfg.val_domain_text_path = Path(val_domain_text_path) if val_domain_text_path else None

    train_text, val_text = resolve_train_val_text(cfg)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer(train_text)
    train_ids = tokenizer.encode(train_text)

    # Validation can contain symbols absent from training vocab; we drop them explicitly.
    val_ids = tokenizer.encode(val_text, drop_unknown=True)

    token_dtype = token_dtype_for_vocab_size(tokenizer.vocab_size)
    tokenizer_path = target_dir / "tokenizer.json"
    train_tokens_path = target_dir / "train.bin"
    val_tokens_path = target_dir / "val.bin"
    metadata_path = target_dir / "meta.json"

    tokenizer.save(tokenizer_path)
    train_num_tokens = write_token_file(train_tokens_path, train_ids, token_dtype)
    val_num_tokens = write_token_file(val_tokens_path, val_ids, token_dtype)

    metadata = TokenizedDatasetMetadata(
        vocab_size=tokenizer.vocab_size,
        token_dtype=token_dtype.name,
        tokenizer_path=tokenizer_path.name,
        train_tokens_path=train_tokens_path.name,
        val_tokens_path=val_tokens_path.name,
        train_num_tokens=train_num_tokens,
        val_num_tokens=val_num_tokens,
        dataset_type=dataset_type,
    )
    metadata.save(metadata_path)
    return metadata


def build_dataloaders(
    config: object,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], CharTokenizer | None]:
    """Builds train/validation dataloaders from a config-like object."""
    tokenizer: CharTokenizer | None = None

    if (
        getattr(config, "token_metadata_path", None) is not None
        or getattr(config, "train_tokens_path", None) is not None
        or getattr(config, "val_tokens_path", None) is not None
    ):
        metadata: TokenizedDatasetMetadata | None = None
        metadata_dir: Path | None = None

        if getattr(config, "token_metadata_path", None) is not None:
            metadata = TokenizedDatasetMetadata.load(config.token_metadata_path)
            metadata_dir = config.token_metadata_path.parent

        if metadata is None and (
            getattr(config, "train_tokens_path", None) is None or getattr(config, "val_tokens_path", None) is None
        ):
            raise ValueError(
                "Token mode requires token_metadata_path, or train_tokens_path + val_tokens_path + token_metadata_path."
            )
        if metadata is None:
            raise ValueError("token_metadata_path is required in token mode")

        train_tokens_path = config.train_tokens_path or metadata_dir / metadata.train_tokens_path
        val_tokens_path = config.val_tokens_path or metadata_dir / metadata.val_tokens_path
        tokenizer_path = metadata_dir / metadata.tokenizer_path

        train_dataset = MemmapTokenDataset(train_tokens_path, config.seq_len, metadata.token_dtype)
        val_dataset = MemmapTokenDataset(val_tokens_path, config.seq_len, metadata.token_dtype)

        if tokenizer_path.exists():
            tokenizer = load_tokenizer(tokenizer_path)

        config.vocab_size = metadata.vocab_size
    else:
        train_text, val_text = resolve_train_val_text(config)
        tokenizer = CharTokenizer(train_text)

        train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        val_ids = torch.tensor(tokenizer.encode(val_text, drop_unknown=True), dtype=torch.long)

        min_tokens = config.seq_len + 1
        if train_ids.numel() < min_tokens:
            raise ValueError(
                "Train corpus too small for seq_len="
                f"{config.seq_len}. Need at least {min_tokens} tokens, got {train_ids.numel()}."
            )
        if val_ids.numel() < min_tokens:
            raise ValueError(
                "Validation corpus too small for seq_len="
                f"{config.seq_len}. Need at least {min_tokens} tokens, got {val_ids.numel()}."
            )

        train_dataset = TextLMDataset(train_ids, config.seq_len)
        val_dataset = TextLMDataset(val_ids, config.seq_len)
        config.vocab_size = tokenizer.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer


def parse_prepare_args() -> argparse.Namespace:
    """Parses CLI args for token-file preparation."""
    parser = argparse.ArgumentParser(description="Prepare dataset into token binary files")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="raw_text",
        choices=["raw_text", "instruction", "domain_text"],
    )

    parser.add_argument("--text_path", type=str, default="")
    parser.add_argument("--train_text_path", type=str, default="")
    parser.add_argument("--val_text_path", type=str, default="")
    parser.add_argument("--train_split", type=float, default=0.9)

    parser.add_argument("--train_instruction_path", type=str, default="")
    parser.add_argument("--val_instruction_path", type=str, default="")
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--input_field", type=str, default="input")
    parser.add_argument("--output_field", type=str, default="output")

    parser.add_argument("--train_domain_text_path", type=str, default="")
    parser.add_argument("--val_domain_text_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for data preparation."""
    args = parse_prepare_args()
    metadata = prepare_token_files(
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        text_path=args.text_path or None,
        train_text_path=args.train_text_path or None,
        val_text_path=args.val_text_path or None,
        train_split=args.train_split,
        train_instruction_path=args.train_instruction_path or None,
        val_instruction_path=args.val_instruction_path or None,
        instruction_field=args.instruction_field,
        input_field=args.input_field,
        output_field=args.output_field,
        train_domain_text_path=args.train_domain_text_path or None,
        val_domain_text_path=args.val_domain_text_path or None,
    )

    print(f"dataset_type={metadata.dataset_type}")
    print(f"vocab_size={metadata.vocab_size}")
    print(f"token_dtype={metadata.token_dtype}")
    print(f"train_num_tokens={metadata.train_num_tokens}")
    print(f"val_num_tokens={metadata.val_num_tokens}")
    print(f"metadata_path={Path(args.output_dir) / 'meta.json'}")


if __name__ == "__main__":
    main()
