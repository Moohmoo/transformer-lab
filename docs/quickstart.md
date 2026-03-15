# Quickstart

## 1. Install Dependencies

```bash
poetry install
```

## 2. Prepare Dataset Tokens

Use the data loader CLI to convert text into `train.bin`, `val.bin`, and `meta.json`.

```bash
python -m src.data.loader \
  --dataset_type raw_text \
  --train_text_path data/train.txt \
  --val_text_path data/val.txt \
  --output_dir data/tokens
```

## 3. Train

```bash
python -m src.training.trainer \
  --token_metadata_path data/tokens/meta.json \
  --checkpoint_dir checkpoints/run1
```

## 4. What To Check After Training

- Logs should print `checkpoint_saved=...` every epoch.
- Best checkpoint is saved as `best_model.pth` when validation improves.
- A short generated text sample is printed at the end when tokenizer data is available.

## 5. Useful Variations

- Use `--dataset_type instruction` for JSON/JSONL instruction datasets.
- Use `--dataset_type domain_text` for specialized text corpora.
- Add `--no_resume` to force a fresh run in an existing checkpoint directory.
