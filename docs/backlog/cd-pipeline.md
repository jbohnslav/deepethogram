# Backlog: Add CD (Continuous Deployment) Pipeline

**Priority:** Medium
**Branch:** cleanup
**Status:** Open

## Goal

Add proper CI/CD to the deepethogram repo so releases, Docker builds, and testing are automated instead of manual.

## What Exists Now

- **Release workflow** (new, on cleanup): Triggers on version bump in pyproject.toml → builds → publishes to PyPI + GitHub Release. Needs PyPI trusted publishing configured.
- **CI workflows**: `main.yml`, `pre-commit.yml` exist. GPU tests (`gpu.yml`) exist but were disabled ("disable gpu tests until I can self-host").
- **Docker images**: 3 Dockerfiles (gui, headless, full) — all NVIDIA CUDA-based, x86 only. Updated for Python 3.11 on cleanup branch.
- **No automated Docker image publishing** (no GHCR/DockerHub push).

## Tasks

### 1. Docker Image CI/CD
- [ ] Add workflow to build Docker images on push/PR (at least verify they build)
- [ ] Add workflow to push images to GHCR on release/tag
- [ ] Consider a non-CUDA Dockerfile for CI validation (CUDA images can't build on ARM runners)

### 2. GPU Test Strategy
- **Problem:** GPU tests (training step, checkpoint save/load) take 20-30 minutes and need actual NVIDIA hardware.
- **Constraint:** Linux GPU box is a dual-boot gaming PC, not always on. Self-hosted runner won't work.
- **Decision: Use Modal for GPU tests.**
  - Modal spins up on-demand GPU instances, runs tests, shuts down.
  - ~$0.50-1.00 per run for 20-30min on a T4/A10G.
  - GitHub Actions workflow calls Modal to provision GPU, run pytest, return results.
  - No always-on hardware needed.
- **Test split:**
  - **Smoke test** (1 training step, <2min) — run on Modal for PRs
  - **Full training test** (20-30min) — run on Modal nightly or pre-release

### 3. PyPI Trusted Publishing
- [ ] Configure on pypi.org: Settings → Publishing → Add GitHub Actions publisher
  - Repository: `jbohnslav/deepethogram`
  - Workflow: `release.yml`
  - Environment: (leave blank or create one)

### 4. Modal GPU Setup
- [ ] Create Modal account and project
- [ ] Write Modal function that: pulls repo, installs deps, runs GPU tests
- [ ] Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as GitHub repo secrets
- [ ] Create GitHub Actions workflow that triggers Modal runs
- [ ] Test with smoke test first, then add full training suite

### 5. CI Modernization
- [ ] Update `main.yml` to test on Python 3.9, 3.10, 3.11
- [x] Use `uv` for CI installs
- [ ] Re-enable GPU tests with self-hosted runner label

## Notes

- Docker builds are x86-only (NVIDIA CUDA base). Can't test locally on ARM Mac Mini.
- The cleanup branch already has the release workflow, Python 3.11 support, and updated Dockerfiles.
- Historic context: GPU tests were manually run on a Linux box. No CI for them ever existed in automated form.

### 6. ARM Docker Images
- [ ] Create ARM-native Dockerfiles (or multi-arch build) so images build natively on ARM Macs and ARM CI runners
- [ ] Current Dockerfiles hardcode `DEG_PLATFORM=linux/amd64` and use x86 Miniconda — unusable on ARM without QEMU emulation (extremely slow)
- [ ] Options:
  - Multi-arch Dockerfiles with platform-conditional Miniconda URL (x86 vs aarch64)
  - Separate `Dockerfile-gui-arm` / `Dockerfile-headless-arm` variants
  - Switch from Miniconda to `uv` for lighter, arch-agnostic installs
- [ ] CUDA images (headless, full) may not have ARM equivalents — investigate NVIDIA ARM support
- [ ] Figure out how ARM and x86 Dockerfiles coexist (build matrix in CI, or single multi-arch Dockerfile with `--platform`)
