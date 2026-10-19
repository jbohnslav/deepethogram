"""Prepare video fixtures and revision-pinned Hugging Face checkpoints for the tests.

Run ``uv sync --dev`` first to install gdown and huggingface_hub.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import gdown
from huggingface_hub import snapshot_download

REPOSITORY_ROOT = Path(__file__).resolve().parent
MODEL_MANIFEST = REPOSITORY_ROOT / "huggingface" / "models.json"
TEST_ARCHIVE = REPOSITORY_ROOT / "tests" / "DATA" / "testing_deepethogram_archive"


def setup_pretrained_models(archive_path: Path) -> None:
    """Download the published checkpoints at immutable revisions and verify their hashes."""
    manifest = json.loads(MODEL_MANIFEST.read_text())
    for model in manifest["models"]:
        destination = archive_path / "models" / "pretrained_models" / model["run_directory"]
        print(f"Preparing {model['name']} from {model['repo_id']}@{model['commit']}")
        snapshot_download(
            repo_id=model["repo_id"],
            revision=model["commit"],
            local_dir=destination,
            allow_patterns=[
                "checkpoint.pt",
                "hyperparameters.yaml",
                "model_config.json",
                "LICENSE",
                "SHA256SUMS",
                "README.md",
            ],
        )
        checksums = (MODEL_MANIFEST.parent / "models" / model["slug"] / "SHA256SUMS").read_text()
        if (destination / "SHA256SUMS").read_text() != checksums:
            raise ValueError(f"Published checksum manifest differs from the checked-in release: {model['repo_id']}")
        for line in checksums.splitlines():
            expected, filename = line.split("  ", 1)
            digest = hashlib.sha256()
            with (destination / filename).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {model['name']}/{filename}")


def setup_tests() -> bool:
    """Set up video fixtures, then always prepare the published pretrained models."""
    try:
        data_dir = TEST_ARCHIVE.parent
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = data_dir / "testing_deepethogram_archive.zip"
        required_items = ["DATA", "models", "project_config.yaml"]

        if all((TEST_ARCHIVE / item).exists() for item in required_items):
            print("Video fixtures already exist. Skipping archive download.")
        else:
            if not zip_path.exists():
                print("Downloading test video archive...")
                gdown.download(id="1IFz4ABXppVxyuhYik8j38k9-Fl9kYKHo", output=str(zip_path), quiet=False)
            print("Extracting test archive...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
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
        return False


if __name__ == "__main__":
    raise SystemExit(0 if setup_tests() else 1)
