import os
import shutil
import time
import platform

# from projects import get_records_from_datadir, fix_config_paths
from deepethogram import projects

this_path = os.path.abspath(__file__)
test_path = os.path.dirname(this_path)
deg_path = os.path.dirname(test_path)

test_data_path = os.path.join(test_path, "DATA")
# the deepethogram test archive should only be read from, never written to
archive_path = os.path.join(test_data_path, "testing_deepethogram_archive")
assert os.path.isdir(archive_path), "{} does not exist!".format(archive_path)
project_path = os.path.join(test_data_path, "testing_deepethogram")
data_path = os.path.join(project_path, "DATA")

config_path = os.path.join(project_path, "project_config.yaml")
config_path_archive = os.path.join(archive_path, "project_config.yaml")
# config_path = os.path.join(project_path, 'project_config.yaml')
cfg_archive = projects.get_config_from_path(archive_path)


def change_to_deepethogram_directory():
    os.chdir(deg_path)


def clean_test_data():
    if not os.path.isdir(project_path):
        return

    # On Windows, we need to handle file permission errors
    if platform.system() == 'Windows':
        max_retries = 3
        for i in range(max_retries):
            try:
                shutil.rmtree(project_path)
                break
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(1)  # Wait a bit for file handles to be released
                    continue
                else:
                    # If we still can't delete after retries, try to ignore errors
                    try:
                        shutil.rmtree(project_path, ignore_errors=True)
                    except:
                        pass  # If we still can't delete, just continue
    else:
        shutil.rmtree(project_path)


def make_project_from_archive():
    change_to_deepethogram_directory()
    clean_test_data()
    shutil.copytree(archive_path, project_path)
    # this also fixes paths
    cfg = projects.get_config_from_path(project_path)
    # projects.fix_config_paths(cfg)


def get_records(origin="project"):
    if origin == "project":
        return projects.get_records_from_datadir(data_path)
    elif origin == "archive":
        return projects.get_records_from_datadir(os.path.join(archive_path, "DATA"))
    else:
        raise NotImplementedError
