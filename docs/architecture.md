# Architecture

The codebase is intentionally split into three top-level concerns.

## `src/data`

Purpose: turn raw datasets into model-ready token windows.

- `tokenizer.py`: `CharTokenizer` and tokenizer save/load helpers.
- `loader.py`: dataset adapters, token-file preparation, and dataloader builders.

## `src/model`

Purpose: assemble a decoder-only transformer from reusable pieces.

- `components/`: low-level blocks (attention, norms, activations, positional).
- `blocks/`: transformer block and FFN block.
- `model_config.py`: architecture options and validation.
- `builder.py`: one entry function to build a validated model.
- `transformer.py`: final model class.

## `src/training`

Purpose: run optimization and checkpointing.

- `config.py`: `TrainConfig` runtime + model settings.
- `checkpoint_store.py`: save/load latest and best checkpoints.
- `trainer.py`: CLI parser, train loop, eval, LR schedule, sample generation.

## Why This Split Helps

- Data changes do not require model refactors.
- Model experiments do not require rewriting the train loop.
- Training behavior stays centralized and easier to debug.
