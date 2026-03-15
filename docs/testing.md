# Testing

## Test Scopes

- Unit: tokenizer, data formatting, config/model validation.
- Integration: dataloaders, checkpoint save/load, mini trainer fit.
- Smoke: tiny end-to-end paths to detect wiring regressions quickly.

## Run All Tests

```bash
poetry run pytest
```

## Run Fast Subset

```bash
poetry run pytest -k "tokenizer or loader"
```

## Lint + Tests Together

```bash
poetry run ruff check src tests
poetry run pytest
```

## CI Expectation

GitHub Actions runs lint and tests on push/pull request so breakages are caught early.
