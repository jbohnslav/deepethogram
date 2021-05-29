import os
import random
import shutil

import numpy as np
import pandas as pd
import pytest

from deepethogram import projects, utils
from deepethogram.data import utils as data_utils
from setup_data import make_project_from_archive, project_path, test_data_path, clean_test_data, get_records


def test_loss_weight():
    class_counts = np.array([1, 2])
    num_pos = np.array([1, 2])
    num_neg = np.array([2, 1])

    pos_weight_transformed, softmax_weight_transformed = data_utils.make_loss_weight(class_counts,
                                                                                     num_pos,
                                                                                     num_neg,
                                                                                     weight_exp=1.0)
    assert np.allclose(pos_weight_transformed, np.array([2, 0.5]))
    assert np.allclose(softmax_weight_transformed, np.array([2 / 3, 1 / 3]))

    class_counts = np.array([0, 300])
    num_pos = np.array([0, 300])
    num_neg = np.array([300, 0])

    pos_weight_transformed, softmax_weight_transformed = data_utils.make_loss_weight(class_counts,
                                                                                     num_pos,
                                                                                     num_neg,
                                                                                     weight_exp=1.0)
    print(pos_weight_transformed, softmax_weight_transformed)
    assert np.allclose(pos_weight_transformed, np.array([0, 1]))
    assert np.allclose(softmax_weight_transformed, np.array([0, 1]))
    # assert np.allclose(pos_weight_transformed, np.array([2, 0.5]))
    # assert np.allclose(softmax_weight_transformed, np.array([2 / 3, 1 / 3]))