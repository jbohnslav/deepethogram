"""Prepare video fixtures and revision-pinned Hugging Face checkpoints for the tests.

Run ``uv sync --dev`` first to install gdown and huggingface_hub.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import traceback
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
RELEASES_DIR = REPOSITORY_ROOT / "huggingface"
MODEL_MANIFEST = RELEASES_DIR / "models.json"
TEST_DATA = REPOSITORY_ROOT / "tests" / "DATA"
TEST_ARCHIVE = TEST_DATA / "testing_deepethogram_archive"
# Downloads live outside the archive, so CI can cache them and re-extracting the archive cannot clobber them.
# DEG_CHECKPOINT_CACHE moves them, e.g. onto the persistent volume the Modal GPU runner mounts.
CHECKPOINT_CACHE = Path(os.environ.get("DEG_CHECKPOINT_CACHE") or TEST_DATA / "huggingface_checkpoints")
RELEASE_FILES = ("checkpoint.pt", "hyperparameters.yaml", "model_config.json", "LICENSE", "SHA256SUMS", "README.md")
# Fields models.json takes from each checked-in model_config.json, which is also the published copy.
CONFIG_FIELDS = ("architecture", "task", "flow_generator")
# Records that a cached release passed verification, so later runs need neither the Hub nor a re-hash.
STAMP_NAME = ".verified.json"


def load_release_manifest(path: Path | None = None) -> dict:
    path = path or MODEL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"Model release manifest not found: {path}. It ships with the repository in huggingface/; "
            "copy that directory next to setup_tests.py."
        )
    manifest = json.loads(path.read_text())
    for model in manifest["models"]:
        config = json.loads((path.parent / "models" / model["slug"] / "model_config.json").read_text())
        model.update({field: config[field] for field in CONFIG_FIELDS})
    return manifest


def checksum_manifest_path(model: dict, releases_dir: Path | None = None) -> Path:
    return (releases_dir or RELEASES_DIR) / "models" / model["slug"] / "SHA256SUMS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_files(directory: Path, manifest_path: Path, skip_hash: tuple[str, ...] = ()) -> None:
    """Raise ValueError unless ``directory`` holds every file in ``manifest_path`` with a matching SHA-256."""
    expected_manifest = manifest_path.read_text()
    published_manifest = directory / "SHA256SUMS"
    if not published_manifest.is_file():
        raise ValueError(f"{directory} has no SHA256SUMS")
    if published_manifest.read_text() != expected_manifest:
        raise ValueError(f"{published_manifest} differs from the checked-in release manifest {manifest_path}")
    for line in expected_manifest.splitlines():
        expected, filename = line.split("  ", 1)
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"{directory} is missing {filename}")
        if filename in skip_hash:
            continue
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}: expected SHA-256 {expected}, got {actual}")


def file_signatures(directory: Path, manifest_text: str) -> dict:
    """Size and modification time of each verified file; raises OSError if one is missing."""
    filenames = ["SHA256SUMS"] + [line.split("  ", 1)[1] for line in manifest_text.splitlines()]
    signatures = {}
    for filename in filenames:
        stat = (directory / filename).stat()
        signatures[filename] = [stat.st_size, stat.st_mtime_ns]
    return signatures


def release_identity(model: dict, manifest_path: Path) -> dict:
    return {"commit": model["commit"], "tag": model["tag"], "sha256sums": manifest_path.read_text()}


def write_stamp(directory: Path, model: dict, manifest_path: Path) -> None:
    stamp = release_identity(model, manifest_path)
    stamp["files"] = file_signatures(directory, stamp["sha256sums"])
    try:
        (directory / STAMP_NAME).write_text(json.dumps(stamp, indent=2))
    except OSError as error:
        # A read-only cache still works; the next run re-hashes instead of trusting the stamp.
        print(f"Could not record verification in {directory}: {error}")


def cached_release_is_current(directory: Path, model: dict, manifest_path: Path) -> bool:
    """True if ``directory`` holds this exact release, checked without contacting the Hub."""
    identity = release_identity(model, manifest_path)
    try:
        stamp = json.loads((directory / STAMP_NAME).read_text())
    except (OSError, ValueError):
        return False
    if {key: stamp.get(key) for key in identity} != identity:
        return False
    try:
        if stamp.get("files") == file_signatures(directory, identity["sha256sums"]):
            return True
        # Files were touched since verification, e.g. restored from a CI cache: re-hash them.
        verify_release_files(directory, manifest_path)
    except (OSError, ValueError):
        return False
    write_stamp(directory, model, manifest_path)
    return True


def check_release_tag(api, model: dict) -> None:
    """The docs tell users to download the model's tag, so it must resolve to the commit CI verifies."""
    resolved = api.model_info(model["repo_id"], revision=model["tag"]).sha
    if resolved != model["commit"]:
        raise ValueError(
            f"{model['repo_id']}@{model['tag']} resolves to {resolved}, but models.json pins {model['commit']}. "
            "Docs and CI would use different weights."
        )


def download_release(model: dict, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model["repo_id"],
        revision=model["commit"],
        local_dir=destination,
        allow_patterns=list(RELEASE_FILES),
    )


def install_release(source: Path, destination: Path) -> None:
    """Copy the release files into ``destination``.

    Copies rather than hard links, so editing a file in the archive cannot change the download cache.
    """
    destination.mkdir(parents=True)
    for filename in RELEASE_FILES:
        if (source / filename).is_file():
            shutil.copy2(source / filename, destination / filename)


def installed_runs_match(pretrained: Path, models: list[dict], cache_dir: Path) -> bool:
    """True if ``pretrained`` holds exactly the release runs, each file an unchanged copy of the cached one."""
    if not pretrained.is_dir():
        return False
    if sorted(path.name for path in pretrained.iterdir()) != sorted(model["run_directory"] for model in models):
        return False
    for model in models:
        source, run = cache_dir / model["slug"], pretrained / model["run_directory"]
        expected = {filename for filename in RELEASE_FILES if (source / filename).is_file()}
        if not run.is_dir() or {path.name for path in run.iterdir()} != expected:
            return False
        for filename in expected:
            cached, installed = (source / filename).stat(), (run / filename).stat()
            if (cached.st_size, cached.st_mtime_ns) != (installed.st_size, installed.st_mtime_ns):
                return False
    return True


def setup_pretrained_models(
    archive_path: Path,
    releases_dir: Path | None = None,
    cache_dir: Path | None = None,
    api=None,
    download=download_release,
) -> None:
    """Install the published checkpoints, at their pinned commits, as the archive's only pretrained models.

    A release already verified in the cache is reused without contacting the Hub. Otherwise its tag is checked,
    the pinned commit downloaded and every file verified. pretrained_models is rebuilt unless it already holds
    exactly these runs, so stale run folders or Lightning checkpoints cannot shadow the pinned weights.
    """
    releases_dir = releases_dir or RELEASES_DIR
    cache_dir = cache_dir or CHECKPOINT_CACHE

    manifest = load_release_manifest(releases_dir / "models.json")
    for model in manifest["models"]:
        directory = cache_dir / model["slug"]
        manifest_path = checksum_manifest_path(model, releases_dir)
        if cached_release_is_current(directory, model, manifest_path):
            print(f"{model['name']}: using verified download in {directory}")
            continue
        print(f"Downloading {model['name']} from {model['repo_id']}@{model['commit']}")
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        check_release_tag(api, model)
        # Start clean, so files from an outdated or corrupt download cannot survive.
        if directory.exists():
            shutil.rmtree(directory)
        download(model, directory)
        verify_release_files(directory, manifest_path)
        write_stamp(directory, model, manifest_path)

    pretrained = archive_path / "models" / "pretrained_models"
    if installed_runs_match(pretrained, manifest["models"], cache_dir):
        print(f"{pretrained} already holds exactly the pinned releases.")
        return
    if pretrained.exists():
        shutil.rmtree(pretrained)
    pretrained.mkdir(parents=True)
    for model in manifest["models"]:
        install_release(cache_dir / model["slug"], pretrained / model["run_directory"])


def setup_tests() -> bool:
    """Set up video fixtures, then prepare the published pretrained models."""
    try:
        import gdown

        TEST_DATA.mkdir(parents=True, exist_ok=True)
        zip_path = TEST_DATA / "testing_deepethogram_archive.zip"
        required_items = ["DATA", "models", "project_config.yaml"]

        if all((TEST_ARCHIVE / item).exists() for item in required_items):
            print("Video fixtures already exist. Skipping archive download.")
        else:
            if not zip_path.exists():
                print("Downloading test video archive...")
                gdown.download(id="1IFz4ABXppVxyuhYik8j38k9-Fl9kYKHo", output=str(zip_path), quiet=False)
            print("Extracting test archive...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(TEST_DATA)
            zip_path.unlink()

        missing_items = [item for item in required_items if not (TEST_ARCHIVE / item).exists()]
        if missing_items:
            raise FileNotFoundError(f"Test archive is missing: {missing_items}")

        # An existing video fixture directory must not bypass checkpoint updates.
        setup_pretrained_models(TEST_ARCHIVE)
        print("Test fixtures and published checkpoints are ready. Run: uv run pytest tests/")
        return True
    except Exception as error:
        print(f"Error during setup: {error}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    raise SystemExit(0 if setup_tests() else 1)
