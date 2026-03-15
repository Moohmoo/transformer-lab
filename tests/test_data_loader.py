import json
from pathlib import Path

import pytest

from src.data.loader import (
    INSTRUCTION_SEPARATOR,
    TokenizedDatasetMetadata,
    build_instruction_text,
    load_json_records,
    prepare_token_files,
    resolve_train_val_text,
)


class DummyConfig:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_load_json_records_json_and_jsonl(tmp_path: Path) -> None:
    json_path = tmp_path / "records.json"
    jsonl_path = tmp_path / "records.jsonl"

    records = [{"instruction": "A", "output": "B"}]
    json_path.write_text(json.dumps(records), encoding="utf-8")
    jsonl_path.write_text('{"instruction": "A", "output": "B"}\n', encoding="utf-8")

    assert load_json_records(json_path) == records
    assert load_json_records(jsonl_path) == records


def test_build_instruction_text_formats_samples() -> None:
    records = [
        {"instruction": "Translate", "input": "hello", "output": "bonjour"},
        {"instruction": "Summarize", "input": "", "output": "short"},
    ]

    text = build_instruction_text(records, "instruction", "input", "output")

    assert "### Instruction:\nTranslate\n" in text
    assert "### Input:\nhello\n" in text
    assert "### Response:\nbonjour\n" in text
    assert INSTRUCTION_SEPARATOR in text


def test_resolve_train_val_text_rejects_invalid_dataset_type() -> None:
    cfg = DummyConfig(dataset_type="bad_type")

    with pytest.raises(ValueError, match="dataset_type"):
        resolve_train_val_text(cfg)


def test_prepare_token_files_and_metadata_roundtrip(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    output = tmp_path / "tokens"

    train.write_text("abcde" * 20, encoding="utf-8")
    val.write_text("abcdeXYZ" * 10, encoding="utf-8")

    metadata = prepare_token_files(
        output_dir=output,
        dataset_type="raw_text",
        train_text_path=train,
        val_text_path=val,
    )

    assert metadata.dataset_type == "raw_text"
    assert metadata.train_num_tokens > 0
    assert metadata.val_num_tokens > 0

    saved = TokenizedDatasetMetadata.load(output / "meta.json")
    assert saved.vocab_size == metadata.vocab_size
    assert (output / saved.train_tokens_path).exists()
    assert (output / saved.val_tokens_path).exists()
    assert (output / saved.tokenizer_path).exists()
