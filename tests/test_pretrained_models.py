"""Check the published checkpoints installed by setup_tests.py, rather than synthetic weights."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from deepethogram import projects, utils
from deepethogram.feature_extractor.models.hidden_two_stream import deg_f, deg_m, deg_s
from deepethogram.flow_generator.models.MotionNet import MotionNet
from deepethogram.flow_generator.models.TinyMotionNet import TinyMotionNet
from deepethogram.flow_generator.models.TinyMotionNet3D import TinyMotionNet3D
from setup_data import archive_path

MODEL_RELEASES = Path(__file__).resolve().parents[1] / "huggingface"
MODELS = json.loads((MODEL_RELEASES / "models.json").read_text())["models"]
PRETRAINED = Path(archive_path) / "models" / "pretrained_models"
CONSTRUCTORS = {
    "tinymotionnet": TinyMotionNet,
    "motionnet": MotionNet,
    "tinymotionnet3d": TinyMotionNet3D,
    "deg-f": lambda: deg_f(num_classes=700, reload_imagenet=False),
    "deg-m": lambda: deg_m(num_classes=700, reload_imagenet=False),
    "deg-s": lambda: deg_s(num_classes=700, reload_imagenet=False),
}


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("release", MODELS, ids=lambda release: release["name"])
def test_published_checkpoint(release):
    directory = PRETRAINED / release["run_directory"]
    assert directory.is_dir(), "Run uv run python setup_tests.py before testing published checkpoints"
    checksums = (MODEL_RELEASES / "models" / release["slug"] / "SHA256SUMS").read_text()
    assert (directory / "SHA256SUMS").read_text() == checksums
    for line in checksums.splitlines():
        expected, filename = line.split("  ", 1)
        digest = hashlib.sha256()
        with (directory / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, f"Wrong published artifact: {release['name']}/{filename}"

    config = json.loads((directory / "model_config.json").read_text())
    shape = [
        1 if dimension == "N" else 64 if dimension in ("H", "W") else dimension for dimension in config["input_shape"]
    ]
    state, _, _ = utils.load_state_dict_from_file(directory / "checkpoint.pt")
    model = CONSTRUCTORS[release["slug"]]()
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.randn(*shape))

    if release["flow_generator"]:
        assert outputs.shape == (1, 700)
        assert torch.isfinite(outputs).all()
    else:
        assert len(outputs) == 3
        for output, size in zip(outputs, (32, 16, 8)):
            expected_shape = (1, 2, 10, size, size) if release["slug"] == "tinymotionnet3d" else (1, 20, size, size)
            assert output.shape == expected_shape
            assert torch.isfinite(output).all()


def test_published_model_discovery():
    discovered = projects.get_weights_from_model_path(PRETRAINED)
    assert set(discovered["flow_generator"]) == {"TinyMotionNet", "MotionNet", "TinyMotionNet3D"}
    assert set(discovered["feature_extractor"]) == {"resnet18", "resnet50", "resnet3d_34"}
    for release in MODELS:
        model_type = "feature_extractor" if release["flow_generator"] else "flow_generator"
        expected = str(PRETRAINED / release["run_directory"] / "checkpoint.pt")
        assert discovered[model_type][release["architecture"]] == [expected]
