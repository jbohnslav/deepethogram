# DeepEthogram Development Setup

DeepEthogram development now uses [uv](https://docs.astral.sh/uv/) end to end. Use Python 3.11 when possible; the project currently supports Python `>=3.9,<3.12`.
The GUI uses PySide6.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
brew install uv
```

## Clone and sync

```bash
git clone https://github.com/jbohnslav/deepethogram.git
cd deepethogram
uv sync
```

## Run the application

```bash
uv run deepethogram
```

## Run tests

If you need the test helpers and extra development tools, sync the dev dependencies first:

```bash
uv sync --dev
uv run python setup_tests.py
uv run pytest tests/
uv run pytest -m gpu
```

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
```

## Optional developer tooling

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Docker

The Docker images are built with uv as well. To build and test the standard targets:

```bash
./docker/build_and_test.sh
```

## Legacy note

If you are maintaining a release prior to 0.4.0 or an older Miniconda/pip environment, keep that work isolated and use the
legacy instructions in [installation.md](installation.md#legacy-installation) instead of mixing old and new environments.
