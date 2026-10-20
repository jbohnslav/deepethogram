"""Check the published checkpoints installed by setup_tests.py, rather than synthetic weights."""

import json

import pytest
import torch

import setup_tests
from deepethogram import projects, utils
from deepethogram.feature_extractor.models.hidden_two_stream import deg_f, deg_m, deg_s
from deepethogram.flow_generator.models.MotionNet import MotionNet
from deepethogram.flow_generator.models.TinyMotionNet import TinyMotionNet
from deepethogram.flow_generator.models.TinyMotionNet3D import TinyMotionNet3D

# A missing manifest fails one test below instead of aborting collection for the whole suite.
try:
    RELEASES = setup_tests.load_release_manifest()["models"]
    MANIFEST_ERROR = None
except FileNotFoundError as error:
    RELEASES = []
    MANIFEST_ERROR = str(error)

PRETRAINED = setup_tests.TEST_ARCHIVE / "models" / "pretrained_models"
MODEL_TYPES = {"optical-flow": "flow_generator", "video-classification": "feature_extractor"}
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


def test_release_manifest_present():
    assert MANIFEST_ERROR is None, MANIFEST_ERROR


@pytest.mark.parametrize("release", RELEASES, ids=lambda release: release["name"])
def test_published_checkpoint(release):
    directory = PRETRAINED / release["run_directory"]
    assert directory.is_dir(), "Run uv run python setup_tests.py before testing published checkpoints"
    # setup_tests.py already hashed the checkpoint; check its identity cheaply here.
    setup_tests.verify_release_files(
        directory, setup_tests.checksum_manifest_path(release), skip_hash=("checkpoint.pt",)
    )
    assert (directory / "checkpoint.pt").stat().st_size == release["checkpoint_bytes"]

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

    if release["task"] == "video-classification":
        assert outputs.shape == (1, 700)
        assert torch.isfinite(outputs).all()
    else:
        assert release["task"] == "optical-flow"
        assert len(outputs) == 3
        for output, size in zip(outputs, (32, 16, 8)):
            expected_shape = (1, 2, 10, size, size) if release["slug"] == "tinymotionnet3d" else (1, 20, size, size)
            assert output.shape == expected_shape
            assert torch.isfinite(output).all()


@pytest.mark.skipif(not RELEASES, reason="model release manifest is missing")
def test_published_model_discovery():
    discovered = projects.get_weights_from_model_path(PRETRAINED)
    for release in RELEASES:
        model_type = MODEL_TYPES[release["task"]]
        expected = str(PRETRAINED / release["run_directory"] / "checkpoint.pt")
        assert discovered[model_type][release["architecture"]] == [expected]
    # Nothing besides the pinned releases may be discoverable.
    for model_type in ("flow_generator", "feature_extractor", "sequence"):
        expected_runs = sum(MODEL_TYPES[release["task"]] == model_type for release in RELEASES)
        assert sum(len(runs) for runs in discovered[model_type].values()) == expected_runs
