# Contributing

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vskaush2/anomaly-detection.git
cd anomaly-detection
uv venv --seed
uv pip install -e ".[dev]"
```

## Running tests

```bash
uv run pytest
```

## Opening a pull request

1. Fork the repository and create a branch from `main`.
2. Make your changes and ensure all tests pass.
3. Open a pull request against `main` — CI will run automatically.
4. A maintainer will review and approve before merging.
