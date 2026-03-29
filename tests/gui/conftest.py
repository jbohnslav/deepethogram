from __future__ import annotations

import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from deepethogram import configuration, projects, utils
from deepethogram.gui import custom_widgets, main as gui_main

CLASS_NAMES = ["background", "walk", "groom"]
FRAME_COUNT = 20
FRAME_SIZE = (64, 48)
FPS = 15.0
WINDOW_TIMEOUT_MS = 10_000
THREAD_TIMEOUT_MS = 1_000
SETTLE_MS = 50


def _write_tiny_video(path: Path, frame_count: int = FRAME_COUNT, offset: int = 0) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic test video at {path}")

    xs = np.linspace(0, 255, FRAME_SIZE[0], dtype=np.uint8)
    ys = np.linspace(0, 255, FRAME_SIZE[1], dtype=np.uint8)
    grid_x = np.tile(xs, (FRAME_SIZE[1], 1))
    grid_y = np.tile(ys[:, None], (1, FRAME_SIZE[0]))

    for frame_index in range(frame_count):
        frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        frame[..., 0] = (grid_x + frame_index * 7 + offset) % 255
        frame[..., 1] = (grid_y + frame_index * 11 + offset) % 255
        frame[..., 2] = (frame_index * 17 + offset) % 255
        writer.write(frame)

    writer.release()
    return path


def _load_project_dict(project_dir: Path) -> dict:
    return utils.load_yaml(project_dir / "project_config.yaml")


def _refresh_record(record_dir: Path) -> None:
    utils.save_dict_to_yaml(projects.parse_subdir(record_dir), record_dir / "record.yaml")


def _record_for_video(videofile: Path) -> dict:
    return projects.get_record_from_subdir(videofile.parent)


def _label_path(videofile: Path) -> Path:
    return videofile.with_name(f"{videofile.stem}_labels.csv")


def _output_path(videofile: Path) -> Path:
    return videofile.with_name(f"{videofile.stem}_outputs.h5")


def _seed_label_csv(path: Path, class_names: list[str], frame_count: int = FRAME_COUNT) -> np.ndarray:
    label_array = np.full((frame_count, len(class_names)), -1, dtype=np.int16)
    label_array[0] = [0, 1, 0]
    label_array[1] = [0, 0, 1]
    label_array[2] = [1, 0, 0]
    pd.DataFrame(label_array, columns=class_names).to_csv(path)
    return label_array


def _seed_output_h5(path: Path, class_names: list[str], frame_count: int = FRAME_COUNT) -> dict[str, np.ndarray]:
    base = np.zeros((frame_count, len(class_names)), dtype=np.float32)
    base[:, 0] = 0.95
    base[3:6, 0] = 0.05
    base[3:6, 1] = 0.92
    base[8:11, 0] = 0.05
    base[8:11, 2] = 0.94

    alt = np.roll(base, 2, axis=0)
    alt[:, 0] = np.where(np.any(alt[:, 1:] > 0.5, axis=1), 0.05, 0.95)
    thresholds = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    names = np.array(class_names, dtype="S")

    with h5py.File(path, "w") as handle:
        for group_name, probabilities in {"resnet18": base, "alt_latent": alt}.items():
            group = handle.create_group(group_name)
            group.create_dataset("P", data=probabilities)
            group.create_dataset("thresholds", data=thresholds)
            group.create_dataset("class_names", data=names)

    return {"resnet18": base, "alt_latent": alt}


def _add_fake_model_runs(project_dir: Path) -> dict[str, Path]:
    model_dir = project_dir / "models"
    runs = {
        "flow_generator": ("240101_000000_flow_generator", "TinyMotionNet"),
        "feature_extractor": ("240101_000100_feature_extractor", "resnet18"),
        "sequence": ("240101_000200_sequence", "tgmj"),
    }
    weight_paths = {}

    for model_name, (run_name, arch) in runs.items():
        run_dir = model_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {"run": {"model": model_name}, model_name: {"arch": arch}}
        if model_name == "sequence":
            config[model_name]["latent_name"] = None
            config[model_name]["output_name"] = None
            config["feature_extractor"] = {"arch": "resnet18"}

        utils.save_dict_to_yaml(config, run_dir / "config.yaml")
        with h5py.File(run_dir / "classification_metrics.h5", "w"):
            pass

        weight_path = run_dir / "checkpoint.pt"
        weight_path.write_text("synthetic checkpoint", encoding="ascii")
        weight_paths[model_name] = weight_path

    return weight_paths


def _wait_for_window(qtbot, window) -> None:
    if not window.isVisible():
        window.show()

    qtbot.waitUntil(window.isVisible, timeout=WINDOW_TIMEOUT_MS)
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        qtbot.waitExposed(window, timeout=WINDOW_TIMEOUT_MS)
        window.raise_()
        window.activateWindow()
        try:
            qtbot.waitUntil(window.isActiveWindow, timeout=WINDOW_TIMEOUT_MS)
        except Exception:
            # Some CI setups are slow to hand off focus even when the widget is usable.
            pass
    qtbot.wait(SETTLE_MS)


def _wait_for_loaded_project(qtbot, window) -> None:
    qtbot.waitUntil(lambda: hasattr(window, "videofile"), timeout=WINDOW_TIMEOUT_MS)
    qtbot.waitUntil(lambda: window.ui.labels.label is not None, timeout=WINDOW_TIMEOUT_MS)
    qtbot.waitUntil(
        lambda: hasattr(window.ui.videoPlayer.videoView, "current_fnum"),
        timeout=WINDOW_TIMEOUT_MS,
    )
    qtbot.waitUntil(lambda: bool(window.ui.nframesLabel.text()), timeout=WINDOW_TIMEOUT_MS)
    _wait_for_window(qtbot, window)


def _terminate_process(process) -> None:
    try:
        if process.poll() is None:
            process.terminate()
        process.wait()
    except Exception:
        pass


def _stop_listener(window) -> None:
    listener = getattr(window, "listener", None)
    if listener is None:
        return

    if hasattr(listener, "pipe"):
        _terminate_process(listener.pipe)
    if hasattr(listener, "stop"):
        listener.stop()
    if hasattr(listener, "should_continue"):
        listener.should_continue = False
    try:
        listener.quit()
    except Exception:
        pass
    try:
        listener.wait(THREAD_TIMEOUT_MS)
    except Exception:
        pass


def _cleanup_window(window) -> None:
    for attr in ("training_pipe", "inference_pipe"):
        process = getattr(window, attr, None)
        if process is not None:
            _terminate_process(process)
            try:
                delattr(window, attr)
            except AttributeError:
                pass

    _stop_listener(window)
    window.saved = True

    if hasattr(window, "vid"):
        try:
            window.vid.close()
        except Exception:
            pass

    for top_level in QtWidgets.QApplication.topLevelWidgets():
        if top_level is window:
            continue
        if isinstance(top_level, QtWidgets.QDialog) and top_level.parent() is window:
            top_level.close()

    window.close()
    window.deleteLater()


@pytest.fixture
def tiny_video_factory(tmp_path):
    def factory(name: str = "tiny_test.avi", *, offset: int = 0, frame_count: int = FRAME_COUNT) -> Path:
        return _write_tiny_video(tmp_path / name, frame_count=frame_count, offset=offset)

    return factory


@pytest.fixture
def tmp_project_dir(tmp_path) -> Path:
    project = projects.initialize_project(
        tmp_path,
        "gui_test",
        behaviors=CLASS_NAMES,
        labeler="tester",
    )
    project["postprocessor"] = {"type": None, "min_bout_length": 1}
    utils.save_dict_to_yaml(project, Path(project["project"]["config_file"]))
    return Path(project["project"]["path"])


@pytest.fixture
def tiny_test_video(tiny_video_factory) -> Path:
    return tiny_video_factory()


@pytest.fixture
def project_with_video(tmp_project_dir: Path, tiny_test_video: Path) -> dict:
    project_dict = _load_project_dict(tmp_project_dir)
    imported_video = Path(projects.add_video_to_project(project_dict, tiny_test_video))
    return {
        "project_dir": tmp_project_dir,
        "project_dict": project_dict,
        "imported_video": imported_video,
        "record_dir": imported_video.parent,
        "record": _record_for_video(imported_video),
        "class_names": CLASS_NAMES,
        "n_frames": FRAME_COUNT,
    }


@pytest.fixture
def project_with_label_csv(project_with_video: dict) -> dict:
    label_path = _label_path(project_with_video["imported_video"])
    label_array = _seed_label_csv(label_path, project_with_video["class_names"], project_with_video["n_frames"])
    _refresh_record(project_with_video["record_dir"])
    project_with_video["label_path"] = label_path
    project_with_video["label_array"] = label_array
    project_with_video["record"] = _record_for_video(project_with_video["imported_video"])
    return project_with_video


@pytest.fixture
def project_with_output_h5(project_with_label_csv: dict) -> dict:
    output_path = _output_path(project_with_label_csv["imported_video"])
    probabilities = _seed_output_h5(output_path, project_with_label_csv["class_names"], project_with_label_csv["n_frames"])
    _refresh_record(project_with_label_csv["record_dir"])
    project_with_label_csv["output_path"] = output_path
    project_with_label_csv["probabilities"] = probabilities
    project_with_label_csv["record"] = _record_for_video(project_with_label_csv["imported_video"])
    return project_with_label_csv


@pytest.fixture
def fake_weights_dir(project_with_output_h5: dict) -> dict[str, Path]:
    return _add_fake_model_runs(project_with_output_h5["project_dir"])


@pytest.fixture
def fake_popen(monkeypatch):
    calls = []

    class FakeProcess:
        def __init__(self, args):
            self.args = args
            self._returncode = 0
            self.terminated = False

        def poll(self):
            return self._returncode

        def terminate(self):
            self.terminated = True
            self._returncode = 0

        def wait(self):
            return self._returncode

    def factory(args, *_, **__):
        process = FakeProcess(args)
        calls.append(process)
        return process

    monkeypatch.setattr(gui_main.subprocess, "Popen", factory)
    monkeypatch.setattr(custom_widgets.subprocess, "Popen", factory)
    return calls


@pytest.fixture
def dialog_monkeypatches(monkeypatch):
    class DialogState:
        existing_directory = ""
        open_file = ("", "")
        open_files = ([], "")
        input_text = ("", False)
        confirm = True
        overwrite = True

    state = DialogState()

    monkeypatch.setattr(gui_main.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: state.existing_directory)
    monkeypatch.setattr(gui_main.QFileDialog, "getOpenFileName", lambda *args, **kwargs: state.open_file)
    monkeypatch.setattr(gui_main.QFileDialog, "getOpenFileNames", lambda *args, **kwargs: state.open_files)
    monkeypatch.setattr(gui_main.QInputDialog, "getText", lambda *args, **kwargs: state.input_text)
    monkeypatch.setattr(gui_main, "simple_popup_question", lambda *args, **kwargs: state.confirm)
    monkeypatch.setattr(gui_main, "overwrite_or_not", lambda *args, **kwargs: state.overwrite)
    return state


@pytest.fixture
def window_factory(monkeypatch, qtbot, tmp_path):
    fake_cwd = tmp_path / "cwd" / "nested"
    fake_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gui_main.os, "getcwd", lambda: str(fake_cwd))
    monkeypatch.setattr(gui_main, "simple_popup_question", lambda *args, **kwargs: True)
    monkeypatch.chdir(fake_cwd)
    created_windows = []

    def factory(project_dir: Path, *, load_project: bool = True):
        cfg = configuration.make_config(
            str(project_dir),
            ["config", "gui", "postprocessor"],
            run_type="gui",
            model=None,
            use_command_line=False,
        )
        window = gui_main.MainWindow(cfg)
        qtbot.addWidget(window)
        _wait_for_window(qtbot, window)
        created_windows.append(window)

        if load_project:
            records = projects.get_records_from_datadir(project_dir / "DATA")
            window.initialize_project(str(project_dir))
            if records:
                _wait_for_loaded_project(qtbot, window)
        return window

    yield factory

    for window in created_windows:
        _cleanup_window(window)

    QtCore.QCoreApplication.processEvents()


@pytest.fixture
def scene_point():
    def factory(view, frame: int, behavior: int):
        point = view.mapFromScene(QtCore.QPointF(frame + 0.5, behavior + 0.5))
        return QtCore.QPoint(int(point.x()), int(point.y()))

    return factory
