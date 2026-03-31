import os
import random
import shutil

import numpy as np
import pandas as pd
import pytest

from deepethogram import projects
from setup_data import make_project_from_archive, project_path, test_data_path, clean_test_data, get_records

# make_project_from_archive()


def test_initialization():
    clean_test_data()

    with pytest.raises(AssertionError):
        project_dict = projects.initialize_project(test_data_path, "testing", ["scratch", "itch"])

    project_dict = projects.initialize_project(test_data_path, "testing", ["background", "scratch", "itch"])
    # print(project_dict)
    # print(project_dict['project'])
    assert os.path.isdir(project_dict["project"]["path"])
    assert project_dict["project"]["path"] == project_path

    data_abs = os.path.join(project_dict["project"]["path"], project_dict["project"]["data_path"])
    assert os.path.isdir(data_abs)

    model_abs = os.path.join(project_dict["project"]["path"], project_dict["project"]["model_path"])
    assert os.path.isdir(model_abs)


# mouse01 tests image directories
@pytest.mark.parametrize("key", ["mouse00", "mouse01"])
def test_add_video(key):
    make_project_from_archive()

    project_dict = projects.load_config(os.path.join(project_path, "project_config.yaml"))
    # project_dict = utils.load_yaml()
    key_path = os.path.join(project_path, "DATA", key)
    assert os.path.isdir(key_path)
    shutil.rmtree(key_path)

    records = get_records("archive")
    # test image directory
    videofile = records[key]["rgb"]
    print(project_dict)
    # this also z-scores, which is pretty slow
    projects.add_video_to_project(project_dict, videofile)
    assert os.path.isdir(os.path.join(project_path, "DATA", key))
    assert os.path.exists(os.path.join(project_path, "DATA", key, os.path.basename(videofile)))


@pytest.mark.parametrize("key", ["mouse00", "mouse01"])
def test_is_deg_file(key):
    make_project_from_archive()
    records = get_records()

    rgb = records[key]["rgb"]
    assert projects.is_deg_file(rgb)

    record_yaml = os.path.join(os.path.dirname(rgb), "record.yaml")
    assert os.path.isfile(record_yaml)
    os.remove(record_yaml)

    assert not projects.is_deg_file(rgb)


def test_add_behavior():
    make_project_from_archive()
    cfg_path = os.path.join(project_path, "project_config.yaml")

    projects.add_behavior_to_project(cfg_path, "A")
    records = get_records()
    mice = list(records.keys())
    labelfile = records[random.choice(mice)]["label"]
    assert os.path.isfile(labelfile)
    df = pd.read_csv(labelfile, index_col=0)
    assert df.shape[1] == 6
    assert np.all(df.iloc[:, -1].values == -1)
    assert df.columns[5] == "A"


def test_remove_behavior():
    make_project_from_archive()
    cfg_path = os.path.join(project_path, "project_config.yaml")
    # can't remove behaviors that don't exist
    with pytest.raises(AssertionError):
        projects.remove_behavior_from_project(cfg_path, "A")

    # can't remove background
    with pytest.raises(ValueError):
        projects.remove_behavior_from_project(cfg_path, "background")

    projects.remove_behavior_from_project(cfg_path, "face_groom")

    records = get_records()
    mice = list(records.keys())
    labelfile = records[random.choice(mice)]["label"]
    assert os.path.isfile(labelfile)
    df = pd.read_csv(labelfile, index_col=0)
    assert df.shape[1] == 4
    assert "face_groom" not in df.columns


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_add_external_label():
    make_project_from_archive()
    mousedir = os.path.join(project_path, "DATA", "mouse06")
    assert os.path.isdir(mousedir), "{} does not exist!".format(mousedir)
    labelfile = os.path.join(mousedir, "labels.csv")
    videofile = os.path.join(mousedir, "mouse06.h5")

    projects.add_label_to_project(labelfile, videofile)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_add_label_deg_style_csv(tmp_path):
    """Test add_label_to_project with DEG-generated CSV (has unnamed index column)."""
    make_project_from_archive()
    mousedir = os.path.join(project_path, "DATA", "mouse06")
    videofile = os.path.join(mousedir, "mouse06.h5")

    # Create a DEG-style CSV with unnamed numeric index
    csv_path = tmp_path / "labels_with_index.csv"
    csv_path.write_text(
        ",background,behavior1,behavior2\n"
        "0,1,0,0\n"
        "1,0,1,0\n"
        "2,0,0,1\n"
    )

    result = projects.add_label_to_project(str(csv_path), videofile)
    df = pd.read_csv(result, index_col=0)
    assert "background" in df.columns
    assert "behavior1" in df.columns
    assert "behavior2" in df.columns
    assert df.shape[1] == 3  # background + 2 behaviors


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_add_label_external_csv_no_index(tmp_path):
    """Test add_label_to_project with external CSV (no index column, no background).

    Regression test for GitHub issue #116: the old code used index_col=0 which
    silently ate the first data column when no index column was present.
    """
    make_project_from_archive()
    mousedir = os.path.join(project_path, "DATA", "mouse06")
    videofile = os.path.join(mousedir, "mouse06.h5")

    # Create a user-provided CSV without index or background column
    csv_path = tmp_path / "labels_no_index.csv"
    csv_path.write_text(
        "behavior1,behavior2\n"
        "0,0\n"
        "1,0\n"
        "0,1\n"
    )

    result = projects.add_label_to_project(str(csv_path), videofile)
    df = pd.read_csv(result, index_col=0)
    assert "background" in df.columns, "background column should be auto-inserted"
    assert "behavior1" in df.columns, "behavior1 should NOT be eaten by index_col"
    assert "behavior2" in df.columns
    assert df.shape[1] == 3  # background + behavior1 + behavior2


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_add_label_external_csv_with_background_no_index(tmp_path):
    """Test external CSV that has background but no index column."""
    make_project_from_archive()
    mousedir = os.path.join(project_path, "DATA", "mouse06")
    videofile = os.path.join(mousedir, "mouse06.h5")

    csv_path = tmp_path / "labels_bg_no_index.csv"
    csv_path.write_text(
        "background,behavior1,behavior2\n"
        "1,0,0\n"
        "0,1,0\n"
        "0,0,1\n"
    )

    result = projects.add_label_to_project(str(csv_path), videofile)
    df = pd.read_csv(result, index_col=0)
    assert "background" in df.columns
    assert "behavior1" in df.columns
    assert "behavior2" in df.columns
    assert df.shape[1] == 3


if __name__ == "__main__":
    test_add_external_label()
