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
CHECKPOINT_CACHE = TEST_DATA / "huggingface_checkpoints"
RELEASE_FILES = ("checkpoint.pt", "hyperparameters.yaml", "model_config.json", "LICENSE", "SHA256SUMS", "README.md")


def load_release_manifest(path: Path | None = None) -> dict:
    path = path or MODEL_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"Model release manifest not found: {path}. It ships with the repository in huggingface/; "
            "copy that directory next to setup_tests.py."
        )
    return json.loads(path.read_text())


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


def check_release_tag(api, model: dict, tag: str) -> None:
    """The docs tell users to download ``tag``, so it must resolve to the commit CI verifies."""
    resolved = api.model_info(model["repo_id"], revision=tag).sha
    if resolved != model["commit"]:
        raise ValueError(
            f"{model['repo_id']}@{tag} resolves to {resolved}, but models.json pins {model['commit']}. "
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
    """Make ``destination`` contain exactly the release files, hard-linked from the download cache when possible."""
    destination.mkdir(parents=True)
    for filename in RELEASE_FILES:
        if not (source / filename).is_file():
            continue
        try:
            os.link(source / filename, destination / filename)
        except OSError:
            shutil.copy2(source / filename, destination / filename)


def setup_pretrained_models(
    archive_path: Path,
    releases_dir: Path | None = None,
    cache_dir: Path | None = None,
    api=None,
    download=download_release,
) -> None:
    """Install the published checkpoints, at their pinned commits, as the archive's only pretrained models.

    The whole pretrained_models directory is rebuilt, so stale run folders or Lightning checkpoints from the
    archive or earlier runs cannot shadow the pinned weights during discovery.
    """
    releases_dir = releases_dir or RELEASES_DIR
    cache_dir = cache_dir or CHECKPOINT_CACHE
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    manifest = load_release_manifest(releases_dir / "models.json")
    for model in manifest["models"]:
        print(f"Preparing {model['name']} from {model['repo_id']}@{model['commit']}")
        check_release_tag(api, model, manifest["release"])
        download(model, cache_dir / model["slug"])
        verify_release_files(cache_dir / model["slug"], checksum_manifest_path(model, releases_dir))

    pretrained = archive_path / "models" / "pretrained_models"
    if pretrained.exists():
        shutil.rmtree(pretrained)
    pretrained.mkdir(parents=True)
    for model in manifest["models"]:
        install_release(cache_dir / model["slug"], pretrained / model["run_directory"])


def setup_tests() -> bool:
    """Set up video fixtures, then always prepare the published pretrained models."""
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
