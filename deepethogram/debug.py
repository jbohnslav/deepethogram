import logging
import os
import pprint
from typing import Union

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from vidio import VideoReader

from deepethogram import file_io, projects

log = logging.getLogger(__name__)


def print_models(model_path: Union[str, os.PathLike]) -> None:
    """Prints all models found in the model path

    Parameters
    ----------
    model_path : Union[str, os.PathLike]
        Absolute path to models directory
    """
    trained_models = projects.get_weights_from_model_path(model_path)
    log.info("Trained models: {}".format(pprint.pformat(trained_models)))


def print_dataset_info(datadir: Union[str, os.PathLike]) -> None:
    """Prints information about your dataset.

    - video path
    - number of unlabeled rows in a video
    - number of examples of each behavior in each video

    Parameters
    ----------
    datadir : Union[str, os.PathLike]
        [description]
    """
    records = projects.get_records_from_datadir(datadir)

    for key, record in records.items():
        log.info("Information about subdir {}".format(key))
        if record["rgb"] is not None:
            log.info("Video: {}".format(record["rgb"]))

        if record["label"] is not None:
            label = file_io.read_labels(record["label"])
            if np.sum(label == -1) > 0:
                unlabeled_rows = np.any(label == -1, axis=0)
                n_unlabeled = np.sum(unlabeled_rows)
                log.warning(
                    "{} UNLABELED ROWS!".format(n_unlabeled)
                    + "VIDEO WILL NOT BE USED FOR FEATURE_EXTRACTOR OR SEQUENCE TRAINING."
                )
            else:
                class_counts = label.sum(axis=0)
                log.info("Labels with counts: {}".format(class_counts))


def try_load_all_frames(datadir: Union[str, os.PathLike]):
    """Attempts to read every image from every video.

    Useful for debugging corrupted videos, e.g. if saving to disk was aborted improperly during acquisition
    If there is an error reading a frame, it will print the video name and frame number

    Parameters
    ----------
    datadir : Union[str, os.PathLike]
        absolute path to the project/DATA directory
    """
    log.info("Iterating through all frames of all movies to test for frame reading bugs")
    records = projects.get_records_from_datadir(datadir)
    for key, record in tqdm(records.items()):
        with VideoReader(record["rgb"]) as reader:
            log.info("reading all frames from file {}".format(record["rgb"]))
            had_error = False
            for i in tqdm(range(len(reader)), leave=False):
                try:
                    _ = reader[i]
                except Exception:
                    had_error = True
                    print("error reading frame {} from video {}".format(i, record["rgb"]))
                except KeyboardInterrupt:
                    raise
            if had_error:
                log.warning("Error in file {}. Is this video corrupted?".format(record["rgb"]))
            else:
                log.info("No problems in {}".format(key))


if __name__ == "__main__":
    if os.path.isfile("debug.log"):
        os.remove("debug.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )

    cfg = OmegaConf.from_cli()
    if cfg.project.path is None and cfg.project.config_file is None:
        raise ValueError("must input either a path or a config file")
    elif cfg.project.path is not None:
        cfg.project.config_file = os.path.join(cfg.project.path, "project_config.yaml")
    elif cfg.project.config_file is not None:
        cfg.project.path = os.path.dirname(cfg.project.config_file)
    else:
        raise ValueError("must input either a path or a config file, not {}".format(cfg))

    assert os.path.isfile(cfg.project.config_file) and os.path.isdir(cfg.project.path)

    user_cfg = OmegaConf.load(cfg.project.config_file)
    cfg = OmegaConf.merge(cfg, user_cfg)
    cfg = projects.convert_config_paths_to_absolute(cfg)

    logging.info(OmegaConf.to_yaml(cfg))

    print_models(cfg.project.model_path)

    print_dataset_info(cfg.project.data_path)

    try_load_all_frames(cfg.project.data_path)
