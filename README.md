<div align="center">

# Transformer Research Lab

<p>Modular PyTorch project for building, training, and experimenting with decoder-only Transformers.</p>

<p>
    <a href="https://github.com/Moohmoo/transformer-lab/actions/workflows/ci.yml">
        <img src="https://github.com/Moohmoo/transformer-lab/actions/workflows/ci.yml/badge.svg" alt="CI" />
    </a>
    <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12" />
    <img src="https://img.shields.io/badge/framework-pytorch-red.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/lint-ruff-46a758.svg" alt="Ruff" />
</p>

</div>

This repository is designed for fast iteration and learning:

- clear separation of `data`, `model`, and `training`
- configurable architecture components (attention, norm, FFN, positional encoding)
- support for multiple dataset families (`raw_text`, `instruction`, `domain_text`)
- practical tooling (tests, linting, CI)

## Table Of Contents

- [Why This Repo](#why-this-repo)
- [Core Features](#core-features)
- [Project Structure](#project-structure)
- [Quickstart (5 Minutes)](#quickstart-5-minutes)
- [Data Workflows](#data-workflows)
- [Training](#training)
- [Common Commands](#common-commands)
- [Quality And CI](#quality-and-ci)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

## Why This Repo

Many Transformer repos are either too minimal for real experimentation or too complex for quick understanding.
This project targets a middle ground: clean architecture, explainable code, and reproducible training workflows.

## Core Features

- decoder-only Transformer built from reusable puzzle pieces
- attention variants: `standard` (`O(n^2)`) and `linear` (`O(n)`)
- normalization variants: `rmsnorm`, `layernorm`
- FFN variants: `swiglu`, `relu`, `gelu`
- positional variants: `rope`, `none`
- dataset adapters for `raw_text`, `instruction`, and `domain_text`
- checkpointing with resume support and best-model tracking

## Project Structure

```text
src/
    data/              # Corpus loading, token prep, dataset adapters
    model/             # Components, blocks, config, builder, transformer
    training/          # Training config, trainer loop, checkpoint store

docs/                # Beginner-friendly documentation
tests/               # Unit, integration, and smoke tests
.github/workflows/   # CI pipeline
```

## Quickstart (5 Minutes)

### 1. Install dependencies

```bash
poetry install
```

### 2. Prepare tokens (example: instruction dataset)

```bash
python -m src.data.loader \
    --dataset_type instruction \
    --train_instruction_path /path/train.jsonl \
    --val_instruction_path /path/val.jsonl \
    --output_dir /path/tokens
```

### 3. Train

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

Add `--no_resume` to force a fresh run in an existing checkpoint folder.

## Data Workflows

### `raw_text`

```bash
python -m src.data.loader \
    --dataset_type raw_text \
    --train_text_path /path/train.txt \
    --val_text_path /path/val.txt \
    --output_dir /path/tokens
```

### `instruction`

Expected record fields are `instruction`, `input`, and `output` by default.
Field names are configurable via CLI (`--instruction_field`, `--input_field`, `--output_field`).

### `domain_text`

Use this for specialized corpora with domain-specific language.
It follows the same plain-text flow as `raw_text` but keeps intent explicit in config/logs.

## Training

Model hyperparameters and runtime options are exposed through CLI flags in `src/training/trainer.py`.

You can customize:

- architecture: `d_model`, `n_heads`, `n_layers`, `attention_type`, `norm_type`, `ffn_type`, `positional_type`
- optimization: `learning_rate`, `weight_decay`, `warmup_steps`, `grad_clip`
- runtime: `batch_size`, `num_epochs`, `seq_len`, checkpoint behavior

## Common Commands

```bash
# Run tests
poetry run pytest

# Lint code
poetry run ruff check src tests

# Prepare token files
python -m src.data.loader --help

# Train model
python -m src.training.trainer --help
```

## Quality And CI

CI workflow: `.github/workflows/ci.yml`

Current pipeline runs:

- `ruff check src tests`
- `pytest`

## Documentation

Start with:

- `docs/README.md`

Then continue with architecture, fundamentals, data, and training guides.

## Contributing

1. Create a branch from `main`.
2. Make focused changes with clear commit messages.
3. Run `poetry run ruff check src tests` and `poetry run pytest`.
4. Open a Pull Request with context and test evidence.

## Troubleshooting

- Push rejected due to GitHub email privacy:
    Set a noreply email in git config:
    `git config --global user.email "<username>@users.noreply.github.com"`
- SSH auth issues:
    Test with `ssh -T git@github.com` and ensure your public key is added in GitHub settings.
- Poetry Python version mismatch:
    Use a compatible interpreter (for example Python 3.12):
    `poetry env use /path/to/python3.12`
