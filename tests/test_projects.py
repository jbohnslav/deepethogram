import os
import shutil

import pytest

from deepethogram import projects
from setup_data import make_project_from_archive, project_path, test_data_path, clean_test_data

# make_project_from_archive()


def test_initialization():
    clean_test_data()

    with pytest.raises(AssertionError):
        project_dict = projects.initialize_project(test_data_path, 'testing', ['scratch', 'itch'])

    project_dict = projects.initialize_project(test_data_path, 'testing', ['background', 'scratch', 'itch'])
    # print(project_dict)
    # print(project_dict['project'])
    assert os.path.isdir(project_dict['project']['path'])
    assert project_dict['project']['path'] == project_path

    data_abs = os.path.join(project_dict['project']['path'], project_dict['project']['data_path'])
    assert os.path.isdir(data_abs)

    model_abs = os.path.join(project_dict['project']['path'], project_dict['project']['model_path'])
    assert os.path.isdir(model_abs)
