from pathlib import Path

import pytest

from src.data.tokenizer import CharTokenizer, load_tokenizer


def test_tokenizer_encode_decode_roundtrip() -> None:
    tokenizer = CharTokenizer("abca")
    token_ids = tokenizer.encode("caba")

    assert tokenizer.decode(token_ids) == "caba"
    assert tokenizer.vocab_size == 3


def test_tokenizer_unknown_character_handling() -> None:
    tokenizer = CharTokenizer("abc")

    with pytest.raises(ValueError, match="Unknown character"):
        tokenizer.encode("abcd")

    assert tokenizer.encode("abcd", drop_unknown=True) == tokenizer.encode("abc")


def test_tokenizer_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        CharTokenizer(["a", "b", "a"])


def test_tokenizer_save_load(tmp_path: Path) -> None:
    tokenizer = CharTokenizer("abcd")
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    restored = load_tokenizer(path)

    assert restored.symbols == tokenizer.symbols
    assert restored.encode("abcd") == tokenizer.encode("abcd")
