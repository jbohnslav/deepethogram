import os
import shutil

this_path = os.path.abspath(__file__)
test_path = os.path.dirname(this_path)

test_data_path = os.path.join(test_path, 'DATA')
# the deepethogram test archive should only be read from, never written to
archive_path = os.path.join(test_data_path, 'testing_deepethogram_archive')
assert os.path.isdir(archive_path)
project_path = os.path.join(test_data_path, 'testing_deepethogram')
data_path = os.path.join(project_path, 'DATA')


def clean_test_data():
    if os.path.isdir(project_path):
        shutil.rmtree(project_path)


def make_project_from_archive():
    clean_test_data()
    shutil.copytree(archive_path, project_path)