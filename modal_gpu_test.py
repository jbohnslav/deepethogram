from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "deepethogram-gpu-tests"
VOLUME_NAME = "deepethogram-test-data"
CUDA_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04"
UV_VERSION = "0.6.14"
DEFAULT_GPU = os.environ.get("DEG_MODAL_GPU", "T4")
REMOTE_WORKDIR = Path("/workspace")
REMOTE_TESTS_DIR = REMOTE_WORKDIR / "tests"
REMOTE_TEST_DATA_DIR = REMOTE_TESTS_DIR / "DATA"
REMOTE_ARCHIVE_DIR = REMOTE_TEST_DATA_DIR / "testing_deepethogram_archive"
VOLUME_MOUNT_PATH = Path("/data")
VOLUME_ARCHIVE_PATH = VOLUME_MOUNT_PATH / "testing_deepethogram_archive"
REMOTE_JUNIT_PATH = Path("/tmp/pytest-gpu.xml")
VENV_BIN_DIR = Path("/opt/venv/bin")

app = modal.App(APP_NAME)
test_data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install(
        "ca-certificates",
        "ffmpeg",
        "libglib2.0-0",
        "libsm6",
        "libx11-6",
        "libxext6",
    )
    .env(
        {
            "UV_PROJECT_ENVIRONMENT": "/opt/venv",
            "UV_PYTHON_DOWNLOADS": "0",
            "UV_LINK_MODE": "copy",
            "UV_COMPILE_BYTECODE": "1",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONPATH": "/workspace:/workspace/tests",
            "DEG_VERSION": "headless",
        }
    )
    .pip_install(f"uv=={UV_VERSION}")
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_WORKDIR / "pyproject.toml"), copy=True)
    .add_local_file("uv.lock", remote_path=str(REMOTE_WORKDIR / "uv.lock"), copy=True)
    .run_commands("cd /workspace && uv sync --frozen --group dev --no-install-project")
    .workdir(str(REMOTE_WORKDIR))
    .add_local_dir(
        "deepethogram",
        remote_path=str(REMOTE_WORKDIR / "deepethogram"),
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        "tests",
        remote_path=str(REMOTE_TESTS_DIR),
        ignore=[
            "DATA",
            "DATA/**",
            "**/__pycache__",
            "**/*.pyc",
        ],
    )
    .add_local_file("pytest.ini", remote_path=str(REMOTE_WORKDIR / "pytest.ini"))
    .add_local_file("setup_tests.py", remote_path=str(REMOTE_WORKDIR / "setup_tests.py"))
)


def _prepare_test_data_copy() -> None:
    REMOTE_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not VOLUME_ARCHIVE_PATH.is_dir():
        raise FileNotFoundError(
            f"Expected Modal volume archive at {VOLUME_ARCHIVE_PATH}, but it was not found. "
            "Upload tests/DATA/testing_deepethogram_archive to the deepethogram-test-data volume first."
        )

    if REMOTE_ARCHIVE_DIR.is_symlink():
        REMOTE_ARCHIVE_DIR.unlink()
    elif REMOTE_ARCHIVE_DIR.exists():
        if REMOTE_ARCHIVE_DIR.is_dir():
            shutil.rmtree(REMOTE_ARCHIVE_DIR)
        else:
            REMOTE_ARCHIVE_DIR.unlink()

    # setup_data.py rewrites paths inside project_config.yaml during import,
    # so the archive must live on writable local storage for the test run.
    shutil.copytree(VOLUME_ARCHIVE_PATH, REMOTE_ARCHIVE_DIR)


def _run_and_stream(command: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PATH"] = f"{VENV_BIN_DIR}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(VENV_BIN_DIR.parent)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output_lines: list[str] = []

    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)

    return process.wait(), "".join(output_lines)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=3 * 60 * 60,
    volumes={str(VOLUME_MOUNT_PATH): test_data_volume.read_only()},
)
def run_gpu_pytest(
    pytest_target: str = "tests/test_integration.py",
    extra_pytest_args: str = "",
) -> dict[str, str | int]:
    _prepare_test_data_copy()
    REMOTE_JUNIT_PATH.unlink(missing_ok=True)

    command = [
        str(VENV_BIN_DIR / "pytest"),
        "-v",
        "-m",
        "gpu",
        pytest_target,
        f"--junitxml={REMOTE_JUNIT_PATH}",
    ]
    if extra_pytest_args:
        command.extend(shlex.split(extra_pytest_args))

    returncode, output = _run_and_stream(command)
    junit_xml = REMOTE_JUNIT_PATH.read_text() if REMOTE_JUNIT_PATH.exists() else ""

    return {
        "command": " ".join(command),
        "returncode": returncode,
        "output": output,
        "junit_xml": junit_xml,
    }


@app.local_entrypoint()
def main(
    pytest_target: str = "tests/test_integration.py",
    extra_pytest_args: str = "",
    write_junit: str = "pytest-gpu.xml",
) -> None:
    result = run_gpu_pytest.remote(
        pytest_target=pytest_target,
        extra_pytest_args=extra_pytest_args,
    )

    junit_xml = str(result.get("junit_xml", ""))
    if write_junit and junit_xml:
        Path(write_junit).write_text(junit_xml)
        print(f"Wrote JUnit XML to {write_junit}")

    print(f"Remote command: {result['command']}")
    print(f"Remote pytest exit code: {result['returncode']}")

    if int(result["returncode"]) != 0:
        raise SystemExit(int(result["returncode"]))
