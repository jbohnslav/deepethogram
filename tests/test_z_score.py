import numpy as np

from deepethogram import projects
from deepethogram.zscore import get_video_statistics
from setup_data import make_project_from_archive, data_path

make_project_from_archive()


def test_single_video():
    records = projects.get_records_from_datadir(data_path)
    videofile = records['mouse00']['rgb']
    stats = get_video_statistics(videofile, 10)
    print(stats)

    mean = np.array([0.010965, 0.02345, 0.0161])
    std = np.array([0.02623, 0.04653, 0.0349])

    assert np.allclose(stats['mean'], mean, rtol=0, atol=1e-4)
    assert np.allclose(stats['std'], std, rtol=0, atol=1e-4)
    assert stats['N'] == 1875000