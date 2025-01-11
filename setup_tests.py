"""This script downloads the test data archive and sets up the testing environment for DeepEthogram.

For it to work, you need to `pip install gdown`
"""

import sys
import zipfile
from pathlib import Path

import gdown
import requests


def download_file(url, destination):
    """Downloads a file from a URL to a destination with progress indication."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    if total_size == 0:
        print("Warning: Content length not provided by server")

    print(f"Downloading to: {destination}")

    with open(destination, "wb") as f:
        downloaded = 0
        for data in response.iter_content(block_size):
            downloaded += len(data)
            f.write(data)

            # Print progress
            if total_size > 0:
                progress = int(50 * downloaded / total_size)
                sys.stdout.write(f"\r[{'=' * progress}{' ' * (50 - progress)}] {downloaded}/{total_size} bytes")
                sys.stdout.flush()
    print("\nDownload completed!")


def setup_tests():
    """Sets up the testing environment for DeepEthogram."""

    # Create tests/DATA directory if it doesn't exist
    tests_dir = Path("tests")
    data_dir = tests_dir / "DATA"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the test archive
    zip_url = "https://drive.google.com/uc?export=download&id=1IFz4ABXppVxyuhYik8j38k9-Fl9kYKHo"
    zip_path = data_dir / "testing_deepethogram_archive.zip"

    try:
        print("Downloading test data archive...")
        gdown.download(id="1IFz4ABXppVxyuhYik8j38k9-Fl9kYKHo", output=str(zip_path), quiet=False)

        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Verify the extraction
        archive_path = data_dir / "testing_deepethogram_archive"
        required_items = ["DATA", "models", "project_config.yaml"]

        missing_items = [item for item in required_items if not (archive_path / item).exists()]

        if missing_items:
            print(f"Warning: The following items are missing: {missing_items}")
            return False

        print("Setup completed successfully!")
        print("\nYou can now run the tests using: pytest tests/")
        print("Note: The zz_commandline test module will take a few minutes to complete.")

        # Clean up the zip file
        zip_path.unlink()
        return True

    except Exception as e:
        print(f"Error during setup: {str(e)}")
        return False


if __name__ == "__main__":
    setup_tests()
