# DeepEthogram Repository Cleanup Plan

## Executive Summary

This document outlines a comprehensive cleanup plan for the DeepEthogram repository to modernize the codebase, improve maintainability, and enhance user experience. The cleanup is organized into phases to ensure systematic improvement without breaking existing functionality.

**Last Updated**: January 2025  
**Current Branch**: cleanup (partially implemented)

## Progress Status

### ✅ Already Completed on Cleanup Branch
- Migrated from setup.py to pyproject.toml
- Consolidated dependencies (removed requirements.txt)
- Simplified installation process
- Added Docker build and test script
- Updated CI/CD workflows
- Added UV package manager support (beta)

## Phase 1: PyTorch Lightning & Installation Fixes (1 week) 🚨

### 1.1 PyTorch Lightning Compatibility (HIGHEST PRIORITY)
**Status**: ❌ Not Started - **Fixes Issues #163, #145, #158**
**Note**: Stay on Python 3.7 to avoid PySide complications
- [ ] Create compatibility layer for Lightning 1.6.5 → 2.x
- [ ] Option A: Pin to intermediate version (1.9.x) that works with Python 3.7
- [ ] Option B: Add compatibility shims:
  - [ ] Detect Lightning version and use appropriate API calls
  - [ ] Wrap trainer instantiation with version checks
  - [ ] Fix `reload_dataloaders_every_epoch` parameter issue
  - [ ] Fix `progress_bar_refresh_rate` parameter issue  
  - [ ] Fix `gpus` vs `accelerator` parameter
  - [ ] Fix FPSCallback `dataloader_idx` parameter
- [ ] Test all training pipelines
- [ ] Document Lightning version requirements

### 1.2 NumPy Compatibility Fix
**Status**: ❌ Not Started - **Fixes Issue #155**
- [ ] Replace all `np.float` with `float` or `np.float64`
- [ ] Replace all `np.int` with `int` or `np.int64`
- [ ] Add numpy version constraint compatible with Python 3.7
- [ ] Test with numpy 1.21.x (last to support Python 3.7 well)

### 1.3 Hydra/OmegaConf Conflict Resolution
**Status**: ❌ Not Started - **Fixes Issue #144**
- [ ] Fix hydra detection in `__init__.py`
- [ ] Ensure omegaconf version compatibility
- [ ] Add clear error message if hydra-core is installed
- [ ] Test installation from clean environment

### 1.4 Installation & Dependency Fixes (Python 3.7 compatible)
**Status**: ⚠️ Build system ready, dependencies not updated
- [ ] Fix scikit-learn 1.0.2 installation for Python 3.7
- [ ] Update dependencies that work with Python 3.7:
  - [ ] pandas to highest version supporting 3.7 (1.3.5)
  - [ ] scikit-learn to 1.0.2 (with proper build deps)
  - [ ] scipy to highest 3.7-compatible version
- [ ] Create requirements-colab.txt for Colab-specific deps
- [ ] Test installation on fresh systems

### 1.5 Critical GUI Fixes (without PySide upgrade)
- [ ] Fix dropdown menu issue (#166) - pretrained weights not selectable
- [ ] Add debugging for Qt platform issues
- [ ] Create platform-specific installation guides
- [ ] Add GUI error recovery mechanisms

## Phase 2: Stabilization and Testing (1-2 weeks)

### 2.1 Test Suite Fixes
**Status**: ⚠️ Docker test script added, but tests may fail
- [ ] Fix all tests broken by dependency updates
- [ ] Add compatibility shims for PyTorch Lightning changes
- [ ] Mock GPU tests for CI/CD without GPU
- [ ] Ensure Docker tests pass for all images
- [ ] Add regression tests for fixed issues

### 2.2 Documentation Updates
**Status**: ⚠️ Some docs added, critical gaps remain
- [ ] Complete CLI documentation (#169, #170)
- [ ] Fill in empty `model_performance.md`
- [ ] Create troubleshooting guide for common issues:
  - [ ] Installation failures by OS
  - [ ] GPU detection problems
  - [ ] Qt/GUI issues
  - [ ] Dependency conflicts
- [ ] Update README with new Python version requirements

### 2.3 Installation Verification
- [ ] Test installation on fresh systems:
  - [ ] Ubuntu 20.04, 22.04
  - [ ] Windows 10, 11
  - [ ] macOS 12, 13, 14
- [ ] Verify Colab notebook works (#173)
- [ ] Test conda environment creation
- [ ] Verify UV installation method

## Phase 3: Python Version Upgrade (2-3 weeks)

### 3.1 Python 3.8+ Migration Planning
**Status**: ❌ Deferred until core issues fixed
**Dependencies**: Requires PySide2 → PySide6 migration
- [ ] Create detailed migration plan for PySide2 → PySide6
- [ ] Identify all Qt-dependent code sections
- [ ] Plan phased migration approach
- [ ] Test PySide6 compatibility on all platforms

### 3.2 Python Version Update
**After PySide6 migration is complete**
- [ ] Update Python constraint to `>=3.8,<3.12`
- [ ] Update all Docker base images
- [ ] Update conda environment
- [ ] Test on Python 3.8, 3.9, 3.10, 3.11

### 3.3 Modern Dependency Updates
**Only after Python 3.8+ is working**
- [ ] Update to latest compatible versions:
  - [ ] pytorch_lightning to 2.x
  - [ ] pandas to 2.x
  - [ ] numpy to 1.24+
  - [ ] scikit-learn to 1.3+
  - [ ] scipy to 1.11+

## Phase 4: Long-term Improvements (3-4 weeks)

### 4.1 Complete PySide6 Migration
**Status**: ❌ Planning needed
- [ ] Create migration plan from PySide2 to PySide6
- [ ] Update all Qt imports and API calls
- [ ] Test on all platforms
- [ ] Update Docker images with new Qt
- [ ] Document any breaking changes

### 4.2 Feature Requests Implementation
- [ ] Batch video selection for inference (#143)
- [ ] Resume training from checkpoint (#149)
- [ ] Better error messages for missing weights
- [ ] Improved model selection UI
- [ ] Add progress bars for long operations

### 4.3 Code Quality and Refactoring
- [ ] Address remaining TODO items
- [ ] Add type hints throughout codebase
- [ ] Improve error handling
- [ ] Refactor configuration system (#1168)
- [ ] Remove redundant parameters (#94)

## Phase 5: Architecture Refactoring (3-4 weeks)

### 5.1 Code Structure Improvements
- [ ] Separate GUI logic from core functionality
- [ ] Create clear API boundaries
- [ ] Implement dependency injection where appropriate
- [ ] Refactor configuration system for clarity

### 5.2 Model Architecture Updates
- [ ] Update model implementations to use latest PyTorch features
- [ ] Implement model registry pattern
- [ ] Add support for custom model architectures
- [ ] Create model zoo with pretrained weights

### 5.3 Plugin System
- [ ] Design plugin architecture for extensions
- [ ] Create plugin API
- [ ] Implement example plugins
- [ ] Document plugin development

## Phase 6: Advanced Features (4-6 weeks)

### 6.1 Workflow Automation
- [ ] Create CLI for batch processing
- [ ] Add experiment tracking (MLflow/W&B integration)
- [ ] Implement automatic hyperparameter tuning
- [ ] Add continuous learning pipeline

### 6.2 Cloud and Deployment
- [ ] Create Docker images for different use cases
- [ ] Add Kubernetes deployment configurations
- [ ] Implement REST API for remote inference
- [ ] Create cloud-friendly storage backends

### 6.3 Extended Functionality
- [ ] Add multi-animal tracking support
- [ ] Implement real-time inference mode
- [ ] Add support for additional video formats
- [ ] Create behavior analysis tools

## Critical Path and Priority Order

### 🔴 MUST DO FIRST (Stay on Python 3.7):
1. **PyTorch Lightning compatibility** - Add shims/version detection
2. **NumPy deprecations** - Fix np.float/np.int usage
3. **Installation fixes** - Hydra conflicts, scikit-learn builds
4. **GUI dropdown fix** - Unblocks workflow

### 🟡 THEN FIX (Still Python 3.7):
1. Colab notebook compatibility
2. Qt platform fixes (workarounds)
3. Documentation completion
4. Testing improvements

### 🟢 FINALLY UPGRADE (Requires planning):
1. PySide2 → PySide6 migration
2. Python 3.8+ support
3. Modern dependency versions
4. Performance optimizations

## Implementation Guidelines

### Quick Wins First
Start with changes that:
- Have minimal risk
- Fix the most user-reported issues
- Can be tested easily
- Don't require major refactoring

### Version Control Strategy
1. **Current branch (cleanup)**: Already has build improvements
2. Create sub-branches for each critical fix
3. Test each fix independently
4. Merge incrementally with thorough testing
5. Tag pre-release versions for testing

### Testing Requirements
For EACH change:
1. Run existing test suite
2. Test on at least 2 OS platforms
3. Verify GUI still works
4. Test training pipeline end-to-end
5. Check Colab compatibility

### Breaking Changes Communication
1. Create migration guide for Lightning 2.x
2. Document Python version requirements clearly
3. Provide compatibility shims where possible
4. Give users warning before major releases

## Success Metrics

### Immediate Success Criteria (Phase 1)
- [ ] Installation works on Python 3.8+ 
- [ ] Colab notebook functional
- [ ] Training runs without Lightning errors
- [ ] GUI dropdowns work
- [ ] 90% of open issues addressed or have workarounds

### Overall Project Health
- [ ] Test coverage > 80%
- [ ] CI/CD passes on all platforms
- [ ] Documentation complete for all features
- [ ] <5 critical bugs reported per month
- [ ] Installation success rate > 95%

## Risk Mitigation

### Potential Risks
1. **Breaking Changes**: Maintain compatibility layer
2. **Performance Regression**: Benchmark before/after
3. **User Disruption**: Provide migration guides
4. **Dependency Conflicts**: Test thoroughly
5. **Data Loss**: Implement backup mechanisms

### Mitigation Strategies
1. Comprehensive testing at each phase
2. Gradual rollout with beta testing
3. Maintain stable branch during development
4. Document all changes thoroughly
5. Provide rollback procedures

## Revised Timeline Based on Current Status

### Already Completed (on cleanup branch)
- ✅ Build system modernization (setup.py → pyproject.toml)
- ✅ Docker improvements
- ✅ Installation simplification

### Immediate Actions (Week 1) - Stay on Python 3.7
- 🔴 Fix PyTorch Lightning compatibility with shims
- 🔴 Fix NumPy deprecations
- 🔴 Fix installation issues (Hydra, scikit-learn)
- 🔴 Fix GUI dropdown bug

### Short Term (Weeks 2-3) - Still Python 3.7
- 🟡 Stabilize all installations
- 🟡 Fix Colab notebook
- 🟡 Complete documentation
- 🟡 Platform-specific fixes

### Medium Term (Weeks 4-6) - Python upgrade
- 🟢 PySide2 → PySide6 migration
- 🟢 Python 3.8+ upgrade
- 🟢 Modern dependency updates

### Long Term (Weeks 7-10)
- 🔵 Performance optimizations
- 🔵 Feature additions
- 🔵 Architecture improvements

### Total: 2 months for critical fixes, 4 months for full modernization

## Immediate Next Steps

1. **Test current cleanup branch thoroughly**
   ```bash
   ./docker/build_and_test.sh  # Already available!
   ```

2. **Create Lightning compatibility branch (Python 3.7)**
   ```bash
   git checkout -b lightning-compat-py37
   # Add version detection in base.py
   # Create compatibility shims
   # Test with Lightning 1.6.5 and 1.9.x
   ```

3. **Fix NumPy and installation issues**
   ```bash
   git checkout -b fix-numpy-install
   # Replace np.float/np.int
   # Fix Hydra detection
   # Test fresh installations
   ```

4. **Fix GUI dropdown bug**
   ```bash
   git checkout -b fix-gui-dropdown
   # Debug pretrained weight selection
   # Test on multiple platforms
   ```

5. **Only after above are stable:**
   ```bash
   git checkout -b pyside6-python38
   # Plan PySide migration first
   # Then upgrade Python version
   ```

## GitHub Issues Resolution Map

| Issue | Fix Location | Priority | Phase | Python Upgrade Required |
|-------|-------------|----------|-------|------------------------|
| #163 (Flow generator) | base.py - Lightning shims | 🔴 Critical | 1 | No |
| #155 (NumPy) | Throughout - np.float | 🔴 Critical | 1 | No |
| #144 (Hydra) | __init__.py | 🔴 Critical | 1 | No |
| #166 (Dropdowns) | gui/main.py | 🔴 Critical | 1 | No |
| #173 (Colab install) | scikit-learn deps | 🟡 High | 2 | Partial |
| #171 (macOS GUI) | Qt workarounds | 🟡 High | 2 | No |
| #164 (Windows Qt) | Platform guide | 🟡 High | 2 | No |
| #172 (Training) | Documentation | 🟢 Medium | 2 | No |

## Key Strategy Change

### Why Stay on Python 3.7 Initially?
- **PySide2 → PySide6 is a MAJOR migration** requiring:
  - Rewriting all Qt imports and many API calls
  - Extensive GUI testing on all platforms  
  - Potentially breaking changes for users
- **Most critical issues can be fixed WITHOUT Python upgrade**:
  - PyTorch Lightning: Use compatibility shims
  - NumPy: Simple find/replace of deprecated calls
  - Installation: Fix dependencies within Python 3.7 constraints

### Phased Approach Benefits
1. **Phase 1**: Fix critical blockers while maintaining stability
2. **Phase 2**: Stabilize and document workarounds
3. **Phase 3**: Plan and execute PySide6 + Python upgrade together
4. **Phase 4**: Modernize with latest dependencies

This approach gets users unblocked FAST while planning the bigger migration carefully.

## Notes

- **Good News**: Build system already modernized on cleanup branch
- **New Priority**: Fix issues WITHOUT Python upgrade first
- **Testing**: Use new Docker test script for validation
- **Communication**: Be clear about phased approach to users