# Python Version Constraint Investigation

Date: 2026-03-28
Branch: `cleanup`

## Executive Summary

The current `requires-python = ">=3.8,<3.9"` does not appear to be justified by the current direct dependency set on `cleanup`.

The cap is historical. It was introduced on 2025-08-11 in commit `06440cb` (`upgrade python; keep old variant`), immediately after earlier work that had pinned the project to Python 3.7 for `PySide2==5.13.2`. The follow-up commit `dae99db` upgraded the GUI stack from `PySide2` to `PySide6`, but did not widen the Python cap again.

For Python 3.9 specifically, none of the direct dependencies investigated here blocks it.

The real picture is:

- Python 3.9: no direct dependency blocker found.
- Python 3.10: likely workable, but `pytorch_lightning==1.6.5` and `scikit-learn<1.1` are no longer on clearly current, officially advertised ground.
- Python 3.11+: `scipy<1.8` is a definite blocker, and `scikit-learn<1.1` is also too old.

## 1. Git History

`requires-python` was changed from `>=3.7,<3.8` to `>=3.8,<3.9` in:

- Commit: `06440cb6356936a0763cb5920ad51cc38cd9a889`
- Date: 2025-08-11
- Message: `upgrade python; keep old variant`

That commit also created `pyproject_py37.toml`, which preserved the old Python 3.7 / PySide2 variant for Docker and legacy installs.

Relevant follow-up:

- Commit: `dae99dba129449bd061204d1ca1f7332bf68ec79`
- Date: 2025-08-11
- Message: `upgrade pyside2 to pyside6`

That commit changed:

- `PySide2==5.13.2` -> `PySide6>=6.6.0`
- Ruff target `py37` -> `py38`

but left `requires-python = ">=3.8,<3.9"` unchanged.

## 2. Original Reason For The Cap

The repo documents the original reason clearly:

- `docs/development_setup.md` says the project was pinned because `PySide2 5.13.2` failed under Python 3.8, and explicitly says "This is why Python was pinned to 3.7".
- `docs/cleanup_plan.md` repeatedly treats Python 3.8+ as blocked on the PySide2 -> PySide6 migration.

Those docs match the old branch state, not the current `cleanup` branch. On `cleanup`, the project already uses `PySide6`.

## 3. Dependency-By-Dependency Findings

### PySide6>=6.6.0

Current lock resolves `PySide6 6.6.3.1`.

- PyPI metadata for `PySide6 6.6.3.1`: `Requires: Python <3.13, >=3.8`
- Classifiers: Python 3.8, 3.9, 3.10, 3.11, 3.12

Conclusion:

- PySide6 is not the blocker anymore.
- It supports Python 3.9+ directly.

### pytorch_lightning==1.6.5

- PyPI metadata for `pytorch-lightning 1.6.5`: `Requires: Python >=3.7`
- Classifiers: Python 3.7, 3.8, 3.9

Conclusion:

- It is not the blocker for Python 3.9.
- It is the main dependency that looks old enough to be a risk for Python 3.10+, but not a proven 3.9 blocker.
- The codebase is tightly coupled to older Lightning APIs:
  - `deepethogram/base.py` uses legacy `Trainer(gpus=...)`
  - `progress_bar_refresh_rate`
  - `reload_dataloaders_every_epoch`
  - direct `self.hparams` assignment
  - legacy Ray Tune integration imports

Practical implication:

- Keeping `1.6.5` is probably fine for a Python 3.9 widening.
- Relaxing Lightning itself is higher-risk than relaxing the scientific stack.
- The repo already anticipated this in `docs/cleanup_plan.md`, which calls for Lightning compatibility work before moving to newer versions.

### pandas<1.4

Current lock resolves `pandas 1.3.5`.

- PyPI metadata for `pandas 1.3.5`: `Requires: Python >=3.7.1`
- Classifiers: Python 3.7, 3.8, 3.9, 3.10
- PyPI metadata for `pandas 2.0.3`: `Requires: Python >=3.8`
- Classifiers: Python 3.8, 3.9, 3.10, 3.11

Code usage in this repo is basic:

- `read_csv`
- `to_csv`
- `DataFrame`
- `concat`
- column rename/reset-index/value access

I did not find use of pandas APIs commonly removed in later versions such as:

- `DataFrame.append`
- `.ix`
- `Panel`
- `as_matrix`

Conclusion:

- `pandas<1.4` does not explain the Python `<3.9` cap.
- I did not find a repo-local code reason for this pin.
- This looks like a legacy compatibility pin, likely from the Python 3.7 era documented in `docs/cleanup_plan.md`.

Risk if relaxed:

- Low for `1.4`/`1.5`.
- Probably still low for `2.0.x`, but that should be validated with tests because pandas 2 tightened a number of long-deprecated behaviors.

### scikit-learn<1.1

Current lock resolves `scikit-learn 1.0.2`.

- PyPI metadata for `scikit-learn 1.0.2`: `Requires: Python >=3.7`
- Classifiers: Python 3.7, 3.8, 3.9
- PyPI metadata for `scikit-learn 1.1.3`: `Requires: Python >=3.8`
- Classifiers: Python 3.8, 3.9, 3.10, 3.11

Code usage in this repo is minimal:

- `f1_score`
- `roc_auc_score`
- `auc`
- `confusion_matrix`

Conclusion:

- This is not a Python 3.9 blocker.
- It is a likely blocker for a cleanly supported Python 3.10/3.11 story, because the pinned version is only explicitly classified through 3.9.
- I did not find any repo-local use of sklearn APIs that would obviously break on 1.1+.

Risk if relaxed:

- Low, based on current usage.

### scipy<1.8

Current lock resolves `scipy 1.7.3`.

- PyPI metadata for `scipy 1.7.3`: `Requires: Python >=3.7, <3.11`
- Classifiers: Python 3.7, 3.8, 3.9, 3.10
- PyPI metadata for `scipy 1.10.1`: `Requires: Python <3.12, >=3.8`
- Classifiers: Python 3.8, 3.9, 3.10, 3.11

Repo usage is tiny:

- one lazy import of `scipy.stats` in `deepethogram/feature_extractor/models/classifiers/inception.py`

Conclusion:

- Not a Python 3.9 blocker.
- Not a Python 3.10 blocker.
- It is a definite blocker for Python 3.11+.

Risk if relaxed:

- Very low from repo code usage alone.

### chardet<4.0

Current lock resolves `chardet 3.0.4`.

- `chardet 3.0.4` project page states: `Requires Python 2.6, 2.7, or 3.3+`
- `chardet 4.0.0` metadata: `Requires: Python >=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*`
- `chardet 5.0.0` metadata: `Requires: Python >=3.6`

Repo usage:

- no direct `chardet` imports found

Conclusion:

- This does not explain the Python `<3.9` cap.
- It may not need to be a direct dependency at all.
- If it is kept, the `<4.0` pin looks unnecessary for Python-version reasons.

Risk if relaxed:

- Very low from repo code inspection.
- Still worth validating install/runtime in case some indirect path expects the old package behavior.

## 4. What Actually Blocks Python 3.9+

Short answer: no direct dependency investigated here blocks Python 3.9.

The current `<3.9` cap appears to be a leftover from the old PySide2 migration path, not a live dependency constraint on `cleanup`.

## 5. What Blocks Higher Versions

### For Python 3.9

No direct blocker found.

### For Python 3.10

No hard blocker found among the direct pins checked here, but two things become questionable:

- `pytorch_lightning==1.6.5` only explicitly advertises Python 3.7-3.9
- `scikit-learn 1.0.2` only explicitly advertises Python 3.7-3.9

So Python 3.10 is more of a support-confidence issue than a clearly impossible environment.

### For Python 3.11

Current pins are too old:

- `scipy<1.8` definitely blocks it
- `scikit-learn<1.1` is also too old
- `pytorch_lightning==1.6.5` is well outside its advertised test matrix by that point

## 6. Suggested Relaxations

### Minimal-risk widening now

If the goal is just to stop artificially rejecting Python 3.9, the evidence supports widening to at least Python 3.9.

Conservative option:

- change Python support to include 3.9
- keep current dependency pins for the first pass
- regenerate `uv.lock`
- run tests on 3.8 and 3.9

### Next most reasonable cleanup

- `pandas<1.4` can likely be relaxed first, with low risk
- `scikit-learn<1.1` can likely be relaxed next, also low risk
- `scipy<1.8` can likely be relaxed next, very low code-risk
- `chardet<4.0` can likely be relaxed or removed

### What not to casually relax

`pytorch_lightning==1.6.5` is the highest-risk pin to change because the codebase is clearly written against older Lightning APIs.

If Lightning is upgraded:

- expect code changes in `deepethogram/base.py`
- expect possible callback/logger/tuner/Ray integration changes
- expect test fallout around training setup

## 7. Non-Dependency Operational Constraints Still In Repo

Even if `pyproject.toml` is widened, several repo files still encode the older world:

- `environment.yml` still pins `python=3.7` and `pyside2=5.13.2`
- Dockerfiles still install Python 3.7 and copy `pyproject_py37.toml`
- `uv.lock` currently has `requires-python = "==3.8.*"`
- docs still describe Python 3.8 as the supported v0.3.0 path

These do not prove a runtime blocker for Python 3.9, but they do mean packaging, docs, and reproducible environments are not yet aligned.

## 8. Recommended Interpretation

The current `requires-python = ">=3.8,<3.9"` is too restrictive for the code and dependencies that are actually on `cleanup`.

My recommendation:

1. Treat Python 3.9 support as unblocked by the direct dependencies reviewed here.
2. Treat `pytorch_lightning==1.6.5` as a future modernization problem, not the reason for `<3.9`.
3. Treat `scipy<1.8` and `scikit-learn<1.1` as the first real blockers once the goal moves to Python 3.10/3.11.
4. Update the operational files (`uv.lock`, Dockerfiles, `environment.yml`, docs) together when the actual widening is performed.

## Sources

- PySide6 6.6.3.1: https://pypi.org/project/PySide6/6.6.3.1/
- pytorch-lightning 1.6.5: https://pypi.org/project/pytorch-lightning/1.6.5/
- pytorch-lightning 1.9.5: https://pypi.org/project/pytorch-lightning/1.9.5/
- pandas 1.3.5: https://pypi.org/project/pandas/1.3.5/
- pandas 2.0.3: https://pypi.org/project/pandas/2.0.3/
- scikit-learn 1.0.2: https://pypi.org/project/scikit-learn/1.0.2/
- scikit-learn 1.1.3: https://pypi.org/project/scikit-learn/1.1.3/
- scipy 1.7.3: https://pypi.org/project/scipy/1.7.3/
- scipy 1.10.1: https://pypi.org/project/scipy/1.10.1/
- chardet 3.0.4: https://pypi.org/project/chardet/3.0.4/
- chardet 4.0.0: https://pypi.org/project/chardet/4.0.0/
- chardet 5.0.0: https://pypi.org/project/chardet/5.0.0/
