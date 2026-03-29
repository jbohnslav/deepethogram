# Installation

DeepEthogram recently migrated from Miniconda/pip to [uv](https://docs.astral.sh/uv/). Current releases should be installed and run with uv.
For versions prior to 0.4.0, see [Legacy Installation](#legacy-installation) below.

## Requirements

- Recommended Python 3.11
- Supported Python `>=3.9,<3.12`
- FFmpeg available on your system
- PySide6 for the GUI
- A recent GPU-enabled PyTorch setup if you plan to train models on CUDA

## Quick Start with uv

1. Install uv.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or
   brew install uv
   ```

   See the [Astral uv documentation](https://docs.astral.sh/uv/) for platform-specific installation details.

2. Install FFmpeg from [ffmpeg.org](https://www.ffmpeg.org/).

3. Clone the repository and sync the development environment.

   ```bash
   git clone https://github.com/jbohnslav/deepethogram.git
   cd deepethogram
   uv sync
   uv run deepethogram
   ```

4. If you only want to install the package as a user, use:

   ```bash
   uv pip install deepethogram
   ```

## Development Setup

For day-to-day development work:

```bash
git clone https://github.com/jbohnslav/deepethogram.git
cd deepethogram
uv sync
uv run deepethogram
```

Useful follow-up commands:

```bash
uv run python setup_tests.py
uv run pytest tests/
uv run ruff check .
uv run ruff format .
```

CI and Docker are uv-based as well, so matching your local workflow to `uv sync` keeps the environment closest to what ships.

## Legacy Installation

Current releases should use uv. Keep the older flow below only for maintaining existing environments or releases prior to 0.4.0.

### Legacy pip install

```bash
python -m pip install deepethogram
```

### Legacy Miniconda / conda install

Older DeepEthogram releases were commonly installed with Miniconda plus pip. If you are maintaining one of those releases,
create a clean environment first, install FFmpeg and PyTorch there, and then install DeepEthogram with pip:

```bash
conda create -n deepethogram python=3.7
conda activate deepethogram
conda install -c conda-forge ffmpeg
python -m pip install deepethogram
```

Legacy releases may still assume PySide2 and older Python constraints, so check the release notes for the exact version you are maintaining.

## Common Problems

### FFmpeg is missing

DeepEthogram uses FFmpeg to read and write `.mp4` files. Install FFmpeg separately and make sure it is available on your `PATH`.

### PySide or Qt import errors

Current DeepEthogram releases use PySide6. In a uv-managed checkout, recreate the environment and resync:

```bash
rm -rf .venv
uv sync
```

If you are troubleshooting an older release that still used PySide2, use the legacy instructions above instead of the current uv workflow.

### OpenCV / Qt plugin issues

Start from a fresh environment first. Most Qt plugin errors are caused by mixing packages from multiple installers in the same environment.
