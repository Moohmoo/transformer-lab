# Data Pipeline

## Supported Dataset Types

- `raw_text`: generic plain text.
- `instruction`: JSON/JSONL records with instruction fields.
- `domain_text`: specialized plain text corpora.

## Preparation Flow

1. Resolve train/validation text from selected dataset adapter.
2. Build char tokenizer from train text.
3. Encode train and validation text.
4. Write compact binary tokens (`train.bin`, `val.bin`).
5. Write metadata (`meta.json`) with dtype, vocab size, and file names.

## Why Validation Unknown Tokens Are Dropped

Tokenizer vocabulary comes from train data only.

During validation, unseen characters are dropped so evaluation can continue without introducing new ids.

## Running In Token Mode

Training can read precomputed token files by setting:

- `--token_metadata_path`
- optional overrides: `--train_tokens_path`, `--val_tokens_path`

This avoids repeated tokenization across experiments.
