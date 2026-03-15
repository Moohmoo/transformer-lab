# Training

## Core Steps

`Trainer.fit` runs this loop per epoch:

1. Train over all batches.
2. Evaluate on validation loader.
3. Save latest checkpoint.
4. Save best checkpoint if validation loss improved.

## Learning Rate Schedule

`get_lr` combines:

- linear warmup for early stability
- cosine decay for smoother late training

## Checkpoint Files

In each run directory:

- `checkpoint_latest.pth`: most recent state.
- `best_model.pth`: best validation model so far.

Stored payload includes:

- model weights
- optimizer state
- epoch and global step
- best validation loss
- serialized config

## Resume Behavior

If `resume=True` and a latest checkpoint exists, training continues from saved state.

Use `--no_resume` to force restart from epoch 0.
