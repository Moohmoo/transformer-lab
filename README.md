# Transformer Research Lab

Modular PyTorch codebase for experimenting with modern Transformer architectures.

## Features

- Decoder-only Transformer assembled from reusable components
- Optional **RoPE** positional embeddings
- Optional **RMSNorm** or **LayerNorm**
- Optional FFN variants: **SwiGLU**, **ReLU**, **GELU**
- Attention variants: **standard** (`O(n^2)`) and **linear** (`O(n)`)
- Data adapters for `raw_text`, `instruction`, and `domain_text`

## Structure

```
src/
├── data/           # Corpus loading, token prep, dataset adapters
├── model/          # Components, blocks, model config, builder, transformer
└── training/       # Train config, trainer loop, checkpoint store

docs/               # Documentation beginner-friendly
tests/              # Unit, integration, smoke tests
.github/workflows/  # CI lint + tests
```

## Installation

```bash
poetry install
```

## Build A Model

```python
from src.model import ModelConfig, build_transformer

config = ModelConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    attention_type="standard",  # or "linear"
)
model = build_transformer(config)
```

## Prepare Data

### Raw Text

```bash
python -m src.data.loader \
    --dataset_type raw_text \
    --train_text_path /path/train.txt \
    --val_text_path /path/val.txt \
    --output_dir /path/tokens
```

### Instruction Dataset (JSON/JSONL)

```bash
python -m src.data.loader \
    --dataset_type instruction \
    --train_instruction_path /path/train.jsonl \
    --val_instruction_path /path/val.jsonl \
    --output_dir /path/tokens
```

## Train

```bash
python -m src.training.trainer \
    --dataset_type instruction \
    --token_metadata_path /path/tokens/meta.json \
    --num_epochs 3 \
    --batch_size 8 \
    --seq_len 64 \
    --max_seq_len 128 \
    --checkpoint_dir ./checkpoints/run1
```

Use `--no_resume` to force a fresh run.

## Dataset Types

- `raw_text`: generic language-modeling text corpora
- `instruction`: JSON/JSONL with `instruction`, `input`, `output` fields
- `domain_text`: specialized plain-text corpora

## Testing

```bash
poetry run pytest
```

## Lint

```bash
poetry run ruff check src tests
```

## CI

GitHub Actions workflow is configured in `.github/workflows/ci.yml`.

It runs:

- `ruff check .`
- `ruff check src tests`
- `pytest`

## Documentation

Start with `docs/README.md`.

## Publish To GitHub

Step-by-step instructions are in `docs/github-publish.md`.

Quick version:

```bash
git init
git branch -M main
git add .
git commit -m "Initial project setup"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
