import os
import shutil

import pytest

from deepethogram import projects

test_dir = os.path.dirname(os.path.abspath(__file__))
data_subdir = os.path.join(test_dir, 'tmp_data')
if os.path.isdir(data_subdir):
    shutil.rmtree(data_subdir)
os.makedirs(data_subdir)


def test_initialization():
    with pytest.raises(AssertionError):
        project_dict = projects.initialize_project(data_subdir, 'testing', ['scratch', 'itch'])

    project_dict = projects.initialize_project(data_subdir, 'testing', ['background', 'scratch', 'itch'])
    # print(project_dict)
    # print(project_dict['project'])
    assert os.path.isdir(project_dict['project']['path'])
    assert project_dict['project']['path'] == os.path.join(data_subdir, 'testing_deepethogram')

    data_abs = os.path.join(project_dict['project']['path'], project_dict['project']['data_path'])
    assert os.path.isdir(data_abs)

    model_abs = os.path.join(project_dict['project']['path'], project_dict['project']['model_path'])
    assert os.path.isdir(model_abs)
