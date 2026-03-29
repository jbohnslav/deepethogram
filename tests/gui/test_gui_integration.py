from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from deepethogram import projects
from deepethogram.gui import main as gui_main
from deepethogram.gui.menus_and_popups import ShouldRunInference

pytestmark = pytest.mark.gui


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _install_fake_inference_dialog(monkeypatch, outputs):
    class FakeShouldRunInference:
        def __init__(self, record_keys, should_start_checked):
            self.record_keys = record_keys
            self.should_start_checked = should_start_checked

        def exec(self):
            return True

        def get_outputs(self):
            return outputs

    monkeypatch.setattr(gui_main, "ShouldRunInference", FakeShouldRunInference)


class _DummyListener:
    def __init__(self, *args, **kwargs):
        self.started = False

    def start(self):
        self.started = True

    def quit(self):
        return None

    def wait(self):
        return None


class _FakeChainer:
    instances = []

    def __init__(self, calls):
        self.calls = calls
        self.started = False
        self.stopped = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def wait(self):
        return None


def test_app_window_boots_and_exposes_major_panels(tmp_project_dir, window_factory):
    window = window_factory(tmp_project_dir, load_project=False)

    assert window.isVisible()
    assert window.ui.videoBox.title() == "Video Info"
    assert window.ui.groupBox.title() == "FlowGenerator"
    assert window.ui.groupBox_2.title() == "FeatureExtractor"
    assert window.ui.groupBox_3.title() == "Sequence"
    assert window.ui.labelBox.title() == "Labels"
    assert window.ui.groupBox_4.title() == "Predictions"
    assert window.ui.videoPlayer is not None
    assert window.ui.labels is not None
    assert window.ui.predictions is not None


def test_project_loads_and_auto_opens_video(project_with_video, window_factory):
    window = window_factory(project_with_video["project_dir"])

    assert Path(window.videofile) == project_with_video["imported_video"]
    assert window.ui.nameLabel.text() == project_with_video["imported_video"].name
    assert window.ui.nframesLabel.text() == str(project_with_video["n_frames"])
    assert window.ui.labels.label.array.shape == (project_with_video["n_frames"], len(project_with_video["class_names"]))
    assert window.ui.predictions.label.array.shape == (
        project_with_video["n_frames"],
        len(project_with_video["class_names"]),
    )


def test_frame_navigation_supports_scrollbar_text_and_shortcuts(project_with_video, qtbot, window_factory):
    window = window_factory(project_with_video["project_dir"])
    timeline = window.ui.videoPlayer.scrollbartext
    viewer = window.ui.videoPlayer.videoView

    timeline.horizontalScrollBar.setValue(5)
    qtbot.waitUntil(lambda: viewer.current_fnum == 5)

    timeline.plainTextEdit.setPlainText("7")
    qtbot.waitUntil(lambda: viewer.current_fnum == 7)

    window.activateWindow()
    window.setFocus()

    qtbot.keyClick(window, QtCore.Qt.Key_Down)
    qtbot.waitUntil(lambda: viewer.current_fnum == 10)

    qtbot.keyClick(window, QtCore.Qt.Key_Up)
    qtbot.waitUntil(lambda: viewer.current_fnum == 7)

    qtbot.keyClick(window, QtCore.Qt.Key_Right)
    qtbot.waitUntil(lambda: viewer.current_fnum == 8)

    qtbot.keyClick(window, QtCore.Qt.Key_Left)
    qtbot.waitUntil(lambda: viewer.current_fnum == 7)

    qtbot.keyClick(window, QtCore.Qt.Key_Right, QtCore.Qt.ControlModifier)
    qtbot.waitUntil(lambda: viewer.current_fnum == project_with_video["n_frames"] - 1)

    qtbot.keyClick(window, QtCore.Qt.Key_Left, QtCore.Qt.ControlModifier)
    qtbot.waitUntil(lambda: viewer.current_fnum == 0)


def test_label_editing_supports_buttons_shortcuts_and_mouse(
    project_with_video,
    qtbot,
    scene_point,
    window_factory,
):
    window = window_factory(project_with_video["project_dir"])
    label_view = window.ui.labels.label

    window.ui.labels.buttons.buttons[1].click()
    qtbot.waitUntil(lambda: label_view.array[0, 1] == 1)

    window.move_n_frames(1)
    qtbot.keyClick(window, QtCore.Qt.Key_2)
    qtbot.waitUntil(lambda: label_view.array[1, 2] == 1)

    click_point = scene_point(label_view, 4, 1)
    qtbot.mouseClick(label_view.viewport(), QtCore.Qt.LeftButton, pos=click_point)

    assert label_view.array[4, 1] == 1
    assert np.all(label_view.array[[0, 1], 0] == 0)


def test_save_writes_csv_with_minus_one_for_untouched_rows(project_with_video, window_factory):
    window = window_factory(project_with_video["project_dir"])

    window.ui.labels.buttons.buttons[1].click()
    window.ui.labels.buttons.buttons[1].click()
    window.move_n_frames(3)
    window.ui.labels.buttons.buttons[2].click()
    window.save()

    label_df = _read_csv(project_with_video["imported_video"].with_name("tiny_test_labels.csv"))
    assert label_df.iloc[0].tolist() == [0, 1, 0]
    assert label_df.iloc[3].tolist() == [0, 0, 1]
    assert label_df.iloc[1].tolist() == [-1, -1, -1]


def test_finalize_converts_untouched_frames_to_background_and_advances(
    dialog_monkeypatches,
    project_with_video,
    tiny_video_factory,
    window_factory,
):
    project_dict = project_with_video["project_dict"]
    second_source = tiny_video_factory("second.avi", offset=80)
    second_video = Path(projects.add_video_to_project(project_dict, second_source))

    window = window_factory(project_with_video["project_dir"])
    initial_video = Path(window.videofile)
    label_path = initial_video.with_name(f"{initial_video.stem}_labels.csv")
    expected_next = second_video if initial_video != second_video else project_with_video["imported_video"]

    dialog_monkeypatches.confirm = True
    window.finalize()

    finalized = _read_csv(label_path)
    assert not np.any(finalized.values == -1)
    assert np.all(finalized["background"].values == 1)
    assert Path(window.videofile) == expected_next


def test_prediction_import_review_export_and_label_bootstrapping(
    dialog_monkeypatches,
    project_with_output_h5,
    qtbot,
    window_factory,
):
    window = window_factory(project_with_output_h5["project_dir"])
    original_labels = _read_csv(project_with_output_h5["label_path"])

    assert window.ui.predictionsCombo.count() == 2
    assert window.latent_name == "resnet18"

    initial_probabilities = window.probabilities.copy()
    window.ui.predictionsCombo.setCurrentText("alt_latent")
    qtbot.waitUntil(lambda: window.latent_name == "alt_latent")
    assert not np.array_equal(initial_probabilities, window.probabilities)

    window.export_predictions()
    prediction_csv = project_with_output_h5["imported_video"].with_name("tiny_test_predictions.csv")
    exported = _read_csv(prediction_csv)
    assert exported.shape == (project_with_output_h5["n_frames"], len(project_with_output_h5["class_names"]))

    dialog_monkeypatches.overwrite = False
    window.import_predictions_as_labels()
    imported = _read_csv(project_with_output_h5["label_path"])
    assert imported.iloc[0].tolist() == original_labels.iloc[0].tolist()
    assert imported.iloc[5].tolist() == window.estimated_labels[5].tolist()

    original_labels.to_csv(project_with_output_h5["label_path"])
    window.import_labelfile(project_with_output_h5["label_path"])
    window.saved = True

    dialog_monkeypatches.overwrite = True
    window.import_predictions_as_labels()
    overwritten = _read_csv(project_with_output_h5["label_path"])
    assert np.array_equal(overwritten.values, window.estimated_labels)


def test_add_and_remove_behavior_dialogs(dialog_monkeypatches, project_with_label_csv, window_factory):
    window = window_factory(project_with_label_csv["project_dir"])
    initial_button_count = len(window.ui.labels.buttons.buttons)

    dialog_monkeypatches.input_text = ("jump", True)
    dialog_monkeypatches.confirm = True
    window.add_class()

    added = _read_csv(project_with_label_csv["label_path"])
    assert "jump" in list(window.cfg.project.class_names)
    assert "jump" in list(added.columns)
    assert np.all(added["jump"].values == -1)
    assert len(window.ui.labels.buttons.buttons) == initial_button_count + 1

    dialog_monkeypatches.input_text = ("jump", True)
    window.remove_class()

    removed = _read_csv(project_with_label_csv["label_path"])
    assert "jump" not in list(window.cfg.project.class_names)
    assert "jump" not in list(removed.columns)
    assert len(window.ui.labels.buttons.buttons) == initial_button_count


def test_batch_inference_selection_dialog_defaults_and_outputs(qtbot):
    dialog = ShouldRunInference(["mouse_a", "mouse_b"], [True, False])
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert dialog.get_outputs() == [True, False]
    dialog.buttons[1].click()
    assert dialog.get_outputs() == [True, True]


def test_train_and_infer_buttons_generate_expected_subprocess_commands(
    fake_popen,
    fake_weights_dir,
    monkeypatch,
    project_with_output_h5,
    window_factory,
):
    _install_fake_inference_dialog(monkeypatch, [True])
    monkeypatch.setattr(gui_main, "UnclickButtonOnPipeCompletion", _DummyListener)

    window = window_factory(project_with_output_h5["project_dir"])
    record_dir = project_with_output_h5["record_dir"]
    expected_sequence = [
        "python",
        "-m",
        "deepethogram.sequence.inference",
        f"project.path={project_with_output_h5['project_dir']}",
        "inference.overwrite=True",
        f"sequence.weights={fake_weights_dir['sequence']}",
        f"inference.directory_list=[{record_dir}]",
    ]

    window.ui.flow_train.setChecked(True)
    window.flow_train()

    window.ui.featureextractor_infer.setChecked(True)
    window.featureextractor_infer()
    assert window.generate_sequence_inference_args() == expected_sequence

    assert fake_popen[0].args == [
        "python",
        "-m",
        "deepethogram.flow_generator.train",
        f"project.path={project_with_output_h5['project_dir']}",
        f"reload.weights={fake_weights_dir['flow_generator']}",
    ]
    assert fake_popen[1].args == [
        "python",
        "-m",
        "deepethogram.feature_extractor.inference",
        f"project.path={project_with_output_h5['project_dir']}",
        "inference.overwrite=True",
        f"feature_extractor.weights={fake_weights_dir['feature_extractor']}",
        f"flow_generator.weights={fake_weights_dir['flow_generator']}",
        f"inference.directory_list=[{record_dir}]",
    ]


def test_classifier_inference_chains_feature_extractor_then_sequence(
    fake_weights_dir,
    monkeypatch,
    project_with_output_h5,
    window_factory,
):
    _FakeChainer.instances.clear()
    _install_fake_inference_dialog(monkeypatch, [True])
    monkeypatch.setattr(gui_main, "SubprocessChainer", _FakeChainer)

    window = window_factory(project_with_output_h5["project_dir"])
    expected_fe = [
        "python",
        "-m",
        "deepethogram.feature_extractor.inference",
        f"project.path={project_with_output_h5['project_dir']}",
        "inference.overwrite=True",
        f"feature_extractor.weights={fake_weights_dir['feature_extractor']}",
        f"flow_generator.weights={fake_weights_dir['flow_generator']}",
        f"inference.directory_list=[{project_with_output_h5['record_dir']}]",
    ]
    expected_sequence = [
        "python",
        "-m",
        "deepethogram.sequence.inference",
        f"project.path={project_with_output_h5['project_dir']}",
        "inference.overwrite=True",
        f"sequence.weights={fake_weights_dir['sequence']}",
        f"inference.directory_list=[{project_with_output_h5['record_dir']}]",
    ]

    window.ui.classifierInference.setChecked(True)
    window.classifier_inference()

    assert len(_FakeChainer.instances) == 1
    assert _FakeChainer.instances[0].started
    assert _FakeChainer.instances[0].calls == [expected_fe, expected_sequence]


def test_run_overnight_chains_flow_feature_extractor_and_sequence(
    fake_weights_dir,
    monkeypatch,
    project_with_output_h5,
    window_factory,
):
    _FakeChainer.instances.clear()
    _install_fake_inference_dialog(monkeypatch, [True])
    monkeypatch.setattr(gui_main, "SubprocessChainer", _FakeChainer)

    window = window_factory(project_with_output_h5["project_dir"])
    expected_flow = [
        "python",
        "-m",
        "deepethogram.flow_generator.train",
        f"project.path={project_with_output_h5['project_dir']}",
        f"reload.weights={fake_weights_dir['flow_generator']}",
    ]
    expected_fe = [
        "python",
        "-m",
        "deepethogram.feature_extractor.inference",
        f"project.path={project_with_output_h5['project_dir']}",
        "inference.overwrite=True",
        f"feature_extractor.weights={fake_weights_dir['feature_extractor']}",
        f"flow_generator.weights={fake_weights_dir['flow_generator']}",
        f"inference.directory_list=[{project_with_output_h5['record_dir']}]",
    ]
    expected_sequence = [
        "python",
        "-m",
        "deepethogram.sequence.inference",
        f"project.path={project_with_output_h5['project_dir']}",
        "inference.overwrite=True",
        f"sequence.weights={fake_weights_dir['sequence']}",
        f"inference.directory_list=[{project_with_output_h5['record_dir']}]",
    ]

    window.ui.actionOvernight.setChecked(True)
    window.run_overnight()

    assert len(_FakeChainer.instances) == 1
    assert _FakeChainer.instances[0].started
    assert _FakeChainer.instances[0].calls == [expected_flow, expected_fe, expected_sequence]
