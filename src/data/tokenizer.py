"""Tokenizer definitions and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CharTokenizer:
    """Character-level tokenizer.

    Args:
        text_or_symbols: Raw corpus text or ordered vocabulary symbols.
    """

    def __init__(self, text_or_symbols: str | list[str]) -> None:
        if isinstance(text_or_symbols, str):
            symbols = sorted(set(text_or_symbols))
        else:
            symbols = list(text_or_symbols)

        if len(symbols) != len(set(symbols)):
            raise ValueError("Tokenizer symbols must be unique")

        self.symbols = symbols
        self.vocab_size = len(symbols)
        self.char_to_idx: dict[str, int] = {char: idx for idx, char in enumerate(symbols)}
        self.idx_to_char: dict[int, str] = {idx: char for idx, char in enumerate(symbols)}

    def encode(self, text: str, drop_unknown: bool = False) -> list[int]:
        """Encodes text to token ids.

        Args:
            text: Input text.
            drop_unknown: Drops unknown symbols instead of raising.

        Returns:
            List of token ids.
        """
        indices: list[int] = []
        for char in text:
            idx = self.char_to_idx.get(char)
            if idx is None:
                if drop_unknown:
                    continue
                raise ValueError(f"Unknown character: {char!r}")
            indices.append(idx)
        return indices

    def decode(self, indices: list[int]) -> str:
        """Decodes token ids to text.

        Args:
            indices: Token ids.

        Returns:
            Decoded string.
        """
        return "".join(self.idx_to_char[idx] for idx in indices)

    def to_dict(self) -> dict[str, Any]:
        """Serializes tokenizer metadata."""
        return {
            "tokenizer_type": "char",
            "symbols": self.symbols,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharTokenizer:
        """Builds tokenizer from serialized metadata."""
        tokenizer_type = payload.get("tokenizer_type")
        if tokenizer_type != "char":
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        symbols = payload.get("symbols")
        if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
            raise ValueError("Invalid tokenizer metadata")

        return cls(symbols)

    def save(self, path: str | Path) -> Path:
        """Saves tokenizer metadata to JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path


def load_tokenizer(path: str | Path) -> CharTokenizer:
    """Loads a tokenizer from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CharTokenizer.from_dict(payload)
