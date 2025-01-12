# Testing DeepEthogram

This document describes how to run and contribute to DeepEthogram's test suite.

## Test Categories

DeepEthogram's tests are divided into two main categories:

1. **Standard Tests**: These include unit tests and basic integration tests that don't require GPU resources. These tests run quickly and are executed by default.

2. **GPU Tests**: These are end-to-end integration tests that require an NVIDIA GPU and significant computational resources. They perform actual model training and inference to ensure the full pipeline works correctly. These tests are marked with the `@pytest.mark.gpu` decorator and are skipped by default.

## Running Tests

### Basic Usage

```bash
# Run all tests except GPU tests (default)
pytest tests/

# Run only GPU tests (requires NVIDIA GPU)
pytest -m gpu

# Run all tests including GPU tests
pytest -m ""
```

### Test Data Setup

Before running tests:

1. Download [`testing_deepethogram_archive.zip`](https://drive.google.com/file/d/1IFz4ABXppVxyuhYik8j38k9-Fl9kYKHo/view?usp=sharing)
2. Create a directory called `DATA` in the tests directory
3. Unzip the archive and move its contents to `deepethogram/tests/DATA/testing_deepethogram_archive/`
4. Verify the path structure: `deepethogram/tests/DATA/testing_deepethogram_archive/{DATA,models,project_config.yaml}`

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
