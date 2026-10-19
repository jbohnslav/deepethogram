"""Regression tests for setup_tests.py, the Docker test images and the release metadata. No network needed."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import setup_tests
from deepethogram import projects

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_TEXT = {
    "hyperparameters.yaml": "run:\n  model: flow_generator\nflow_generator:\n  arch: {arch}\n",
    "model_config.json": '{{"architecture": "{arch}"}}\n',
    "LICENSE": "license\n",
}


class FakeHub:
    """Stands in for both HfApi and snapshot_download."""

    def __init__(self, models, tag_commits=None, corrupt=()):
        self.models = {model["slug"]: model for model in models}
        self.tag_commits = tag_commits or {}
        self.corrupt = set(corrupt)
        self.downloads = []

    def model_info(self, repo_id, revision):
        model = next(model for model in self.models.values() if model["repo_id"] == repo_id)
        return SimpleNamespace(sha=self.tag_commits.get(model["slug"], model["commit"]))

    def download(self, model, destination):
        self.downloads.append(model["slug"])
        destination.mkdir(parents=True, exist_ok=True)
        for filename, content in release_files(model).items():
            (destination / filename).write_bytes(content)
        if model["slug"] in self.corrupt:
            (destination / "checkpoint.pt").write_bytes(b"truncated")


def release_files(model):
    files = {name: text.format(arch=model["architecture"]).encode() for name, text in RELEASE_TEXT.items()}
    files["checkpoint.pt"] = f"weights for {model['slug']}".encode()
    sums = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in sorted(files.items()))
    files["SHA256SUMS"] = sums.encode()
    return files


@pytest.fixture
def releases(tmp_path):
    """A releases directory shaped like huggingface/, with two flow generators."""
    models = [
        {
            "slug": slug,
            "name": arch,
            "repo_id": f"owner/{slug}",
            "architecture": arch,
            "task": "optical-flow",
            "commit": f"{index}" * 40,
            "run_directory": f"20010{index}_000000_{arch}",
        }
        for index, (slug, arch) in enumerate([("tinymotionnet", "TinyMotionNet"), ("motionnet", "MotionNet")], 1)
    ]
    releases_dir = tmp_path / "huggingface"
    for model in models:
        model_dir = releases_dir / "models" / model["slug"]
        model_dir.mkdir(parents=True)
        (model_dir / "SHA256SUMS").write_bytes(release_files(model)["SHA256SUMS"])
    (releases_dir / "models.json").write_text(json.dumps({"release": "v1.0", "models": models}))
    return releases_dir, models


def write_run(directory, arch, weights=b"stale"):
    directory.mkdir(parents=True)
    (directory / "hyperparameters.yaml").write_text(RELEASE_TEXT["hyperparameters.yaml"].format(arch=arch))
    (directory / "checkpoint.pt").write_bytes(weights)


def run_setup(tmp_path, releases_dir, hub):
    archive = tmp_path / "archive"
    setup_tests.setup_pretrained_models(
        archive, releases_dir=releases_dir, cache_dir=tmp_path / "cache", api=hub, download=hub.download
    )
    return archive / "models" / "pretrained_models"


def test_stale_runs_cannot_shadow_pinned_checkpoints(tmp_path, releases):
    releases_dir, models = releases
    pretrained = tmp_path / "archive" / "models" / "pretrained_models"
    pinned = pretrained / models[0]["run_directory"]
    # An older run of the same architecture, as the Google Drive archive can contain.
    write_run(pretrained / "190101_000000_TinyMotionNet", "TinyMotionNet")
    # Leftovers inside a pinned run folder: a Lightning checkpoint wins over checkpoint.pt in discovery.
    write_run(pinned, "TinyMotionNet")
    (pinned / "lightning_checkpoints").mkdir()
    (pinned / "lightning_checkpoints" / "epoch=3.ckpt").write_bytes(b"stale")
    (pinned / "notes.txt").write_text("stale")

    pretrained = run_setup(tmp_path, releases_dir, FakeHub(models))

    assert sorted(path.name for path in pretrained.iterdir()) == sorted(model["run_directory"] for model in models)
    for model in models:
        run = pretrained / model["run_directory"]
        assert {path.name for path in run.iterdir()} == set(release_files(model))
        setup_tests.verify_release_files(run, setup_tests.checksum_manifest_path(model, releases_dir))
    discovered = projects.get_weights_from_model_path(pretrained)
    for model in models:
        expected = str(pretrained / model["run_directory"] / "checkpoint.pt")
        assert discovered["flow_generator"][model["architecture"]] == [expected]


def test_rerun_reuses_cache_and_reinstalls(tmp_path, releases):
    releases_dir, models = releases
    hub = FakeHub(models)
    pretrained = run_setup(tmp_path, releases_dir, hub)
    (pretrained / "extra_run").mkdir()

    pretrained = run_setup(tmp_path, releases_dir, hub)

    assert not (pretrained / "extra_run").exists()
    assert (tmp_path / "cache" / "tinymotionnet" / "checkpoint.pt").is_file()


def test_corrupt_download_fails_with_digests(tmp_path, releases):
    releases_dir, models = releases
    with pytest.raises(ValueError, match=r"Checksum mismatch .*checkpoint\.pt: expected SHA-256 [0-9a-f]{64}, got"):
        run_setup(tmp_path, releases_dir, FakeHub(models, corrupt={"motionnet"}))


def test_release_tag_must_resolve_to_pinned_commit(tmp_path, releases):
    releases_dir, models = releases
    hub = FakeHub(models, tag_commits={"motionnet": "f" * 40})
    with pytest.raises(ValueError, match="owner/motionnet@v1.0 resolves to f+, but models.json pins"):
        run_setup(tmp_path, releases_dir, hub)


def test_setup_tests_reports_missing_manifest(tmp_path, monkeypatch, capsys):
    """Without huggingface/ next to it, setup must fail clearly rather than with a bare traceback."""
    fixtures = tmp_path / "DATA"
    archive = fixtures / "testing_deepethogram_archive"
    for item in ("DATA", "models"):
        (archive / item).mkdir(parents=True)
    (archive / "project_config.yaml").write_text("")
    monkeypatch.setattr(setup_tests, "TEST_DATA", fixtures)
    monkeypatch.setattr(setup_tests, "TEST_ARCHIVE", archive)
    monkeypatch.setattr(setup_tests, "RELEASES_DIR", tmp_path / "missing")
    monkeypatch.setattr(setup_tests, "MODEL_MANIFEST", tmp_path / "missing" / "models.json")

    assert setup_tests.setup_tests() is False
    assert "Model release manifest not found" in capsys.readouterr().out


def test_pretrained_model_tests_collect_without_manifest(tmp_path, monkeypatch):
    """A missing manifest must fail one test, not abort collection of the whole suite."""
    monkeypatch.setattr(setup_tests, "MODEL_MANIFEST", tmp_path / "missing.json")
    path = REPOSITORY_ROOT / "tests" / "test_pretrained_models.py"
    spec = importlib.util.spec_from_file_location("pretrained_models_without_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.RELEASES == []
    with pytest.raises(AssertionError, match="Model release manifest not found"):
        module.test_release_manifest_present()


def dockerfile_stages():
    stages, name = {}, None
    for line in (REPOSITORY_ROOT / "docker" / "Dockerfile").read_text().splitlines():
        match = re.match(r"FROM\s+\S+\s+AS\s+(\S+)", line, re.IGNORECASE)
        if match:
            name = match.group(1)
            stages[name] = []
        elif name:
            stages[name].append(line)
    return stages


# The Docker build files are not copied into the test images themselves.
requires_docker_files = pytest.mark.skipif(
    not (REPOSITORY_ROOT / ".dockerignore").is_file() or not (REPOSITORY_ROOT / "docker" / "Dockerfile").is_file(),
    reason="Docker build files are not in this checkout",
)


@requires_docker_files
@pytest.mark.parametrize("stage", ["test-gui", "test-headless", "test-full"])
def test_docker_test_images_include_release_manifests(stage):
    lines = dockerfile_stages()[stage]
    assert any(re.match(r"COPY\s+huggingface\s+\./huggingface/?$", line) for line in lines)
    assert any(re.match(r"COPY\s+.*\bsetup_tests\.py\b", line) for line in lines)


@requires_docker_files
def test_dockerignore_allows_release_manifests():
    rules = [line.strip() for line in (REPOSITORY_ROOT / ".dockerignore").read_text().splitlines()]
    assert "!huggingface/**" in rules
    # A later exclusion would win over the allow rule.
    allowed_at = rules.index("!huggingface/**")
    assert not any(rule.startswith("huggingface") for rule in rules[allowed_at + 1 :])


# test_pretrained_models.py reports a missing manifest; don't abort collection here.
RELEASES_ON_DISK = setup_tests.load_release_manifest()["models"] if setup_tests.MODEL_MANIFEST.is_file() else []


@pytest.mark.parametrize("release", RELEASES_ON_DISK, ids=lambda release: release["slug"])
def test_manifest_matches_model_config(release):
    config_path = setup_tests.RELEASES_DIR / "models" / release["slug"] / "model_config.json"
    config = json.loads(config_path.read_text())
    for field in ("architecture", "task", "flow_generator"):
        assert release[field] == config[field], f"models.json and {config_path} disagree on {field}"
    assert release["task"] in ("optical-flow", "video-classification")


@pytest.mark.parametrize("release", RELEASES_ON_DISK, ids=lambda release: release["slug"])
def test_model_card_metadata(release):
    card = (setup_tests.RELEASES_DIR / "models" / release["slug"] / "README.md").read_text()
    front_matter = card.split("---")[1]
    tags = re.findall(r"^- (\S+)$", front_matter.split("tags:")[1], re.MULTILINE)
    assert len(tags) == len(set(tags)), f"duplicate tags: {tags}"
    assert not re.search(r"\ba (?=[aeiou])", card), "use 'an' before a vowel"
