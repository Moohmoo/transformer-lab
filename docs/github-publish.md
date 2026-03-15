# Publish To GitHub

This project currently appears to be outside a Git repository in your workspace tools.

Use these commands from the project root:

```bash
cd /home/mohamed/M2DATA/transformer-lab

git init
git branch -M main
git add .
git commit -m "Initial clean architecture, docs, tests, and CI"

git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If the remote already exists:

```bash
git remote -v
# If needed, replace URL:
git remote set-url origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Recommended First Checks Before Push

```bash
/home/mohamed/miniconda3/bin/python3.12 -m compileall src tests
/home/mohamed/miniconda3/bin/python3.12 -m pytest -q
```

If you use Poetry with Python 3.12:

```bash
poetry env use /home/mohamed/miniconda3/bin/python3.12
poetry install
poetry run ruff check src tests
poetry run pytest
```
