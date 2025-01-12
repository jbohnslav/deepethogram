# this is named test__zz_commandline so that it comes last, after all module-specific tests
import subprocess
import pytest

from deepethogram import utils

from setup_data import make_project_from_archive, change_to_deepethogram_directory, config_path, data_path
# from setup_data import get_testing_directory

# testing_directory = get_testing_directory()
# config_path = os.path.join(testing_directory, 'project_config.yaml')
BATCH_SIZE = 2  # small but not too small
# if less than 10, might have bugs with visualization
STEPS_PER_EPOCH = 20
NUM_EPOCHS = 2
VIZ_EXAMPLES = 2

make_project_from_archive()

change_to_deepethogram_directory()


def command_from_string(string):
    command = string.split(" ")
    if command[-1] == "":
        command = command[:-1]
    print(command)
    return command


def add_default_arguments(string, train=True):
    string += f"project.config_file={config_path} "
    string += f"compute.batch_size={BATCH_SIZE} "
    if train:
        string += f"train.steps_per_epoch.train={STEPS_PER_EPOCH} train.steps_per_epoch.val={STEPS_PER_EPOCH} "
        string += f"train.steps_per_epoch.test={STEPS_PER_EPOCH} "
        string += f"train.num_epochs={NUM_EPOCHS} "
        string += f"train.viz_examples={VIZ_EXAMPLES}"
    return string


# def test_python():
#     command = ['which', 'python']
#     ret = subprocess.run(command)

#     command = ['which', 'pytest']
#     ret = subprocess.run(command)
#     # assert ret.returncode == 0
#     # print(ret)

#     print(os.environ['PATH'])
#     print(os.getcwd())


@pytest.mark.gpu
def test_flow():
    make_project_from_archive()
    string = "python -m deepethogram.flow_generator.train preset=deg_f "
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    string = "python -m deepethogram.flow_generator.train preset=deg_m "
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    string = "python -m deepethogram.flow_generator.train preset=deg_s "
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.gpu
def test_feature_extractor():
    string = "python -m deepethogram.feature_extractor.train preset=deg_f flow_generator.weights=latest "
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    string = "python -m deepethogram.feature_extractor.train preset=deg_m flow_generator.weights=latest "
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    # for resnet3d, must specify weights, because we can't just download them from the torchvision repo
    string = (
        "python -m deepethogram.feature_extractor.train preset=deg_s flow_generator.weights=latest "
        "feature_extractor.weights=latest "
    )
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    # testing softmax
    string = (
        "python -m deepethogram.feature_extractor.train preset=deg_m flow_generator.weights=latest "
        "feature_extractor.final_activation=softmax "
    )
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.gpu
def test_feature_extraction(softmax: bool = False):
    # the reason for this complexity is that I don't want to run inference on all directories
    string = (
        "python -m deepethogram.feature_extractor.inference preset=deg_f feature_extractor.weights=latest "
        "flow_generator.weights=latest "
    )
    if softmax:
        string += "feature_extractor.final_activation=softmax "
    # datadir = os.path.join(testing_directory, 'DATA')
    subdirs = utils.get_subfiles(data_path, "directory")
    # np.random.seed(42)
    # subdirs = np.random.choice(subdirs, size=100, replace=False)
    dir_string = ",".join([str(i) for i in subdirs])
    dir_string = "[" + dir_string + "]"
    string += f"inference.directory_list={dir_string} inference.overwrite=True "
    string = add_default_arguments(string, train=False)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0
    # string += 'inference.directory_list=[]'


@pytest.mark.gpu
def test_sequence_train():
    string = "python -m deepethogram.sequence.train "
    string = add_default_arguments(string)
    command = command_from_string(string)
    print(command)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    # mutually exclusive
    string = "python -m deepethogram.sequence.train feature_extractor.final_activation=softmax "
    string = add_default_arguments(string)
    command = command_from_string(string)
    print(command)
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.gpu
def test_softmax():
    make_project_from_archive()
    string = "python -m deepethogram.flow_generator.train preset=deg_f "
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    string = (
        "python -m deepethogram.feature_extractor.train preset=deg_f flow_generator.weights=latest "
        "feature_extractor.final_activation=softmax "
    )
    string = add_default_arguments(string)
    command = command_from_string(string)
    ret = subprocess.run(command)
    assert ret.returncode == 0

    test_feature_extraction(softmax=True)

    string = "python -m deepethogram.sequence.train feature_extractor.final_activation=softmax "
    string = add_default_arguments(string)
    command = command_from_string(string)
    print(command)
    ret = subprocess.run(command)
    assert ret.returncode == 0


if __name__ == "__main__":
    test_softmax()
