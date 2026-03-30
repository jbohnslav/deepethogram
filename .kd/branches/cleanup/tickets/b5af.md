---
id: "b5af"
status: closed
deps: []
links: []
created: 2026-03-29T15:39:57Z
type: feature
priority: 2
closed_at: 2026-03-30T14:51:07Z
parent: 4681
tags: [docker, arm, ci]
---
# ARM-native Docker images

Current Dockerfiles hardcode linux/amd64 and use x86 Miniconda — unusable on ARM without QEMU emulation (extremely slow). Need ARM-native or multi-arch Docker builds.

Options:
- Multi-arch Dockerfiles with platform-conditional Miniconda URL (x86 vs aarch64)
- Separate Dockerfile-*-arm variants
- Switch from Miniconda to uv for lighter, arch-agnostic installs
- CUDA images (headless, full) may not have ARM equivalents — investigate NVIDIA ARM support
- Figure out how ARM and x86 coexist (build matrix in CI, or single multi-arch Dockerfile with --platform)

## Acceptance Criteria

- [ ]

**Note (2026-03-29 15:53):** Decision: use uv instead of Miniconda in all Docker images. Lighter, faster, no conda bloat. This applies to both ARM and x86 images.

## Worklog

- [2026-03-30 10:51] — Completed and merged to cleanup in PR #174. GUI target builds on ARM via cpu-only base.
