# Testing DeepEthogram

This document describes how to run and contribute to DeepEthogram's test suite. CI uses uv, and local test runs should do the same.

## Test Categories

DeepEthogram's tests are divided into two main categories:

1. **Standard Tests**: These include unit tests and basic integration tests that don't require GPU resources. These tests run quickly and are executed by default.

2. **GPU Tests**: These are end-to-end integration tests that require an NVIDIA GPU and significant computational resources. They perform actual model training and inference to ensure the full pipeline works correctly. These tests are marked with the `@pytest.mark.gpu` decorator and are skipped by default.

## Running Tests

Sync the development environment first:

```bash
uv sync --dev
uv run python setup_tests.py
```

### Basic Usage

```bash
# Run all tests except GPU tests (default)
uv run pytest tests/

# Run only GPU tests (requires NVIDIA GPU)
uv run pytest -m gpu

# Run all tests including GPU tests
uv run pytest -m ""
```

### Test Data Setup

`uv run python setup_tests.py` prepares everything the tests need:

1. It downloads the video fixture archive from Google Drive into `tests/DATA/testing_deepethogram_archive/`, unless that folder already has `DATA`, `models` and `project_config.yaml`.
2. It downloads the six published checkpoints from Hugging Face at the commits pinned in `huggingface/models.json`, into `tests/DATA/huggingface_checkpoints/`. It checks each model's release tag resolves to its pinned commit, and verifies every file against the checksums in `huggingface/models/<model>/SHA256SUMS`.
3. It copies those six runs into `tests/DATA/testing_deepethogram_archive/models/pretrained_models/`, removing anything else there.

Re-run it after pulling changes to `huggingface/models.json`. Once the downloads are verified, later runs reuse them without contacting Hugging Face, so setup also works offline. Don't unzip the archive by hand: its bundled checkpoints are older snapshots without the release manifests, and `tests/test_pretrained_models.py` fails on them.

## Writing Tests

### Adding GPU Tests

When writing tests that require GPU resources:

1. Mark the test with the `@pytest.mark.gpu` decorator
2. Place GPU-intensive tests in appropriate test modules
3. Keep GPU tests focused and efficient to minimize resource usage

Example:
```python
import pytest

@pytest.mark.gpu
def test_model_training():
    # GPU-intensive test code here
    pass
```

### Best Practices

1. Keep GPU tests separate from standard tests when possible
2. Document resource requirements in test docstrings
3. Use small datasets and minimal epochs for GPU tests
4. Add appropriate error handling for cases where GPU is not available

## Continuous Integration

The CI pipeline runs standard tests by default. GPU tests are only run in specific environments or when explicitly requested to avoid unnecessary resource usage.
