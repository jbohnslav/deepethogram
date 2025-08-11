# DeepEthogram Development Setup and Test Results

## Setup Date: January 2025

## System Information
- **OS**: Linux 5.19.5-051905-generic
- **GPUs**: 2x NVIDIA GeForce RTX 3090 (24GB VRAM each)
- **Python**: System has 3.10.12, project requires 3.7

## Development Environment Setup

### Using UV Package Manager
Successfully set up with UV (v0.8.8) but with a critical modification:
- **Changed Python requirement** from `>=3.7,<3.8` to `>=3.8,<3.9` temporarily
- This allows UV to manage the environment (UV doesn't support Python 3.7)
- Created `.venv` with Python 3.8.20

### Installation Command
```bash
# After modifying pyproject.toml to allow Python 3.8
uv sync
```

### Created Files
- `.venv/` - Virtual environment with Python 3.8.20
- `uv.lock` - Locked dependencies for reproducible builds

## Test Results Summary

### Non-GUI Tests: ✅ PASSING (12/12)
```bash
source .venv/bin/activate && pytest tests/ -v --ignore=tests/test_gui.py
```

All core functionality tests pass:
- `test_data.py::test_loss_weight` ✅
- `test_flow_generator.py::test_metrics` ✅
- `test_models.py::test_get_cnn` ✅
- `test_projects.py` (8 tests) ✅
- `test_z_score.py::test_single_video` ✅

### GUI Tests: ❌ FAILING (0/1)
```bash
tests/test_gui.py::test_setup FAILED
```
**Error**: `TypeError: 'Shiboken.ObjectType' object is not iterable`
- Location: `deepethogram/gui/custom_widgets.py:176`
- Cause: PySide2 5.13.2 incompatibility with Python 3.8
- **This is why Python was pinned to 3.7**

### GPU Tests: ⚠️ TIMEOUT
- Integration tests with GPU (`test_integration.py`) start but timeout after 2 minutes
- `test_flow` passes but other tests hang

## Key Issues Identified

### 1. Python Version Constraint
- **Root Cause**: PySide2 5.13.2 requires Python 3.7
- **Impact**: Cannot use modern Python (3.8+) without migrating to PySide6
- **Current Workaround**: Using Python 3.8 for non-GUI development

### 2. Test Warnings
- PyTorch FutureWarning about `weights_only=False` in torch.load
- Kornia deprecation warning about `torch.cuda.amp.custom_fwd`
- Both are upstream dependency issues, not critical

### 3. Development vs Production
- Development can proceed with Python 3.8 for core functionality
- GUI development still requires Python 3.7 or PySide migration
- Docker tests can validate Python 3.7 compatibility

## Recommendations

### Immediate Actions
1. **For Core Development**: Continue with Python 3.8 setup
2. **For GUI Testing**: Use Docker with Python 3.7
3. **For CI/CD**: Keep dual testing (UV for core, Docker for full)

### Medium-term Solutions
1. **Create compatibility shims** for PyTorch Lightning
2. **Fix NumPy deprecations** (np.float, np.int)
3. **Investigate PySide2 fixes** for Python 3.8 or plan PySide6 migration

### Testing Strategy
```bash
# Core tests (Python 3.8 with UV)
source .venv/bin/activate
pytest tests/ --ignore=tests/test_gui.py

# Full tests including GUI (Python 3.7 with Docker)
./docker/build_and_test.sh
```

## Docker Setup and Testing

### Docker Images Built
Three Docker images have been successfully built and tested with Python 3.7:

1. **deepethogram:headless** - GPU support, no GUI (25.4GB)
   - ✅ nvidia-smi working
   - ✅ CUDA available in PyTorch
   - ✅ All tests pass

2. **deepethogram:gui** - CPU-only, GUI support (13.5GB)
   - ✅ GUI imports and runs successfully
   - ✅ Uses pip-installed PyTorch 1.11.0+cpu to avoid conda/Python 3.7 compatibility issues
   - ✅ GUI window displays correctly with X11 forwarding

3. **deepethogram:full** - GPU + GUI support (built from Dockerfile-full)
   - ✅ nvidia-smi working
   - ✅ CUDA available in PyTorch (1.13.1+cu117)
   - ✅ GUI imports and runs successfully
   - ✅ Detects both RTX 3090 GPUs

### Key Docker Findings

#### Python 3.7 + PyTorch Compatibility Issue
- **Problem**: Conda-installed PyTorch (1.13.1, 1.12.1, 1.11.0) with CPU-only builds fail on Python 3.7
- **Error**: `undefined symbol: iJIT_NotifyEvent` in libtorch_cpu.so
- **Solution**: Install PyTorch via pip instead of conda for CPU-only builds
- **Applied to**: Dockerfile-gui now uses:
  ```dockerfile
  RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  ```

#### GPU Support Verification
Both headless and full containers successfully:
- Access both RTX 3090 GPUs
- Show correct CUDA version (11.5.2 in container, 12.1 driver)
- Report CUDA available in PyTorch
- Can run GPU-accelerated computations

#### GUI Testing with X11
To run GUI from Docker:
```bash
# Allow Docker X11 access
xhost +local:docker

# Run GUI with display forwarding
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --net=host deepethogram:gui deepethogram

# For GPU-enabled GUI
docker run --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --net=host deepethogram:full deepethogram
```

### Docker Test Commands
```bash
# Build and test all images
./docker/build_and_test.sh

# Test GPU access
docker run --rm --gpus all deepethogram:headless nvidia-smi
docker run --rm --gpus all deepethogram:full python -c "import torch; print(torch.cuda.is_available())"

# Test GUI
docker run --rm deepethogram:gui python -c "from deepethogram.gui import main; print('GUI works!')"
```

## Next Steps
1. ~~Run Docker tests to validate Python 3.7 compatibility~~ ✅ Complete
2. ~~Investigate PySide2/Python 3.8 compatibility fixes~~ ✅ Root cause identified
3. Begin implementing PyTorch Lightning compatibility shims
4. Consider migrating from PySide2 to PySide6 for Python 3.8+ support