# Installation

DeepEthogram v0.3.0+ is documented as a uv-managed project. Use uv first, and only fall back to pip or conda if you are maintaining an older legacy environment.

## Requirements

- Python 3.9 to 3.11
- FFmpeg available on your system
- A recent GPU-enabled PyTorch setup if you plan to train models on CUDA
- PySide6 for the GUI (current releases no longer use PySide2)

## Quick Start with uv

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install FFmpeg from [ffmpeg.org](https://www.ffmpeg.org/).

3. Clone the repository and sync the project environment.

   ```bash
   git clone https://github.com/jbohnslav/deepethogram.git
   cd deepethogram
   uv sync
   ```

4. Start the GUI.

   ```bash
   uv run deepethogram
   ```

## Install from PyPI with uv

If you only want to run the application, use an ephemeral tool environment:

```bash
uvx --from deepethogram deepethogram
```

If you are already inside another uv-managed project and want `deepethogram` as a dependency, add it with:

```bash
uv add deepethogram
```

## Development and Tests

Use the `dev` dependency group when you need test data helpers, `pytest`, or `pre-commit`:

```bash
uv sync --dev
uv run python setup_tests.py
uv run pytest tests/
```

## Legacy Fallback

If you are maintaining an older non-uv environment, pip still works as a fallback, but it is no longer the primary installation path:

```bash
python -m pip install deepethogram
```

If you use conda, create a clean environment first, install FFmpeg and PyTorch there, then install DeepEthogram with pip as the last step.

## Common Problems

### FFmpeg is missing

DeepEthogram uses FFmpeg to read and write `.mp4` files. Install FFmpeg separately and make sure it is available on your `PATH`.

### PySide or Qt import errors

Current DeepEthogram releases use PySide6. In a uv-managed checkout, recreate the environment and resync:

```bash
rm -rf .venv
uv sync
```

If you are troubleshooting an older legacy release that still used PySide2, follow that release's historical install notes instead of the current uv workflow.

### OpenCV / Qt plugin issues

Start from a fresh environment first. Most Qt plugin errors are caused by mixing packages from multiple installers in the same environment.
