# ML Fundamentals

## Logits

A transformer outputs raw scores called logits with shape `(batch, seq_len, vocab_size)`.

- Higher logit for a token means the model currently prefers that token.
- `cross_entropy` compares logits against the true next token.

## Why Shifted Targets

Language modeling predicts the next token.

- Input window: `tokens[t : t+seq_len]`
- Target window: `tokens[t+1 : t+seq_len+1]`

This one-step shift teaches autoregressive prediction.

## Layers vs Blocks

In this project, one transformer block contains:

1. pre-norm
2. self-attention
3. residual add
4. pre-norm
5. feed-forward network
6. residual add

Stacking blocks builds depth (`n_layers`).

## Perplexity

Perplexity is derived from validation loss:

- lower is better
- roughly indicates how uncertain the model is on next-token prediction

## Instruction Data vs Raw Text

- `raw_text`: model learns from plain next-token continuation.
- `instruction`: records are formatted into a stable prompt/response template before tokenization.

Instruction formatting helps supervised fine-tuning by making task structure explicit.
