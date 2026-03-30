---
id: "7bce"
status: open
deps: []
links: []
created: 2026-03-29T17:17:32Z
type: task
priority: 3
tags: [packaging, deps, uv]
---
# Dependency group split: gui vs headless

Currently all deps are in one flat [project.dependencies] including PySide6. DEG_VERSION env var controls runtime behavior but packaging doesn't split.

Design question: Want the INVERSE of typical optional deps pattern:
- Default install (pip install deepethogram) = FULL with GUI, everything most users need
- Optional HEADLESS mode that excludes PySide6 for server/cluster use

This is inverted from the usual 'pip install foo[gui]' pattern. Need to figure out how uv/pip handle this. Options:
- Default deps include PySide6, optional 'headless' extra that's a subset (unclear if pip supports this)
- Separate packages (deepethogram, deepethogram-headless) 
- Environment marker approach
- Just keep DEG_VERSION runtime check and install everything everywhere (current approach, works but wastes space)

Research how uv and pip handle 'install less by default, more with extras' vs the typical 'install more with extras' pattern.

## Acceptance Criteria

- [ ]

**Note (2026-03-29 17:18):** Important: research should be uv-native, not pip-based. pip is legacy — use uv's own packaging/install/extras system for everything. No pip in the final solution.
