import logging
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def interpolate_bad_values(keypoint: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Interpolates keypoints with low confidence

    Parameters
    ----------
    keypoint : np.ndarray
        [T, n_keypoints, 3] array. Last dimension: height, width, confidence
    threshold : float, optional
        Values below this threshold will be linearly interpolated, by default 0.9

    Returns
    -------
    np.ndarray
        Keypoint array with height and width interpolated if conf < threshold
    """
    assert keypoint.ndim == 3
    T, kp, _ = keypoint.shape
    assert keypoint.shape[2] == 3

    keypoint_interped = keypoint.copy()

    log.debug("fraction of points below {:.1f}: {:.4f}".format(threshold, np.mean(keypoint[..., 2] < threshold)))
    # TODO: VECTORIZE
    for i in range(kp):
        for j in range(2):
            is_bad = keypoint[:, i, j] < threshold
            # don't want to deal with extrapolation
            is_bad[0] = False
            is_bad[-1] = False

            if is_bad.sum() == 0:
                continue
            # print(i, j, is_bad.sum())
            bad_inds = np.where(is_bad)[0]
            good_inds = np.where(np.logical_not(is_bad))[0]
            y = keypoint[good_inds, i, j]
            x_interp = np.interp(bad_inds, good_inds, y)
            keypoint_interped[bad_inds, i, j] = x_interp
    return keypoint_interped


def normalize_keypoints(keypoints: np.ndarray, H: int, W: int) -> np.ndarray:
    """Normalizes keypoints from range [(0, H), (0, W)] to range [(-1, 1), (-1, 1)].
    Non-square images will use the maximum side length in the denominator.

    Parameters
    ----------
    keypoints : np.ndarray
        [T, n_keypoints, 2] array. Last dimension: height, width
    H : int
        Image height
    W : int
        Image width

    Returns
    -------
    np.ndarray
        Normalized keypoint array
    """
    side_length = max(H, W)
    keypoints = 2 * ((keypoints / side_length) - 0.5)

    return keypoints


def denormalize_keypoints(keypoints, H, W):
    """Un-normalizes keypoints from range [(-1, 1), (-1, 1)] to range [(0, H), (0, W)].
    Non-square images will use the maximum side length.

    Parameters
    ----------
    keypoints : np.ndarray
        [T, n_keypoints, 2] array. Last dimension: height, width
    H : int
        Image height
    W : int
        Image width

    Returns
    -------
    np.ndarray
        Un-normalized keypoint array
    """
    side_length = max(H, W)
    keypoints = (keypoints * 0.5 + 0.5) * side_length
    return keypoints


def slow_alignment(keypoints, rotmats, origins):
    aligned = []
    for i in range(len(keypoints)):
        keypoint = keypoints[i]
        sub = keypoint - origins[i]
        # print(rotmats[i].shape, sub.shape)
        aligned.append((rotmats[i] @ sub.T).T)
    aligned = np.stack(aligned)
    return aligned


def compute_distance(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Computes euclidean distance along the final dimension of two input arrays.

    Parameters
    ----------
    arr1 : np.ndarray
    arr2 : np.ndarray

    Returns
    -------
    np.ndarray
        distance
    """
    return np.sqrt(((arr1 - arr2) ** 2).sum(axis=-1))


def poly_area(x: np.ndarray, y: np.ndarray):
    """Returns area of the polygon specified by X and Y coordinates.
    REQUIRES POINTS TO BE IN CLOCKWISE ORDER!!

    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Parameters
    ----------
    x : np.ndarray
        x position
    y : np.ndarray
        y position

    Returns
    -------
    float
        Area of the polygon
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def load_dlcfile(dlcfile: Union[str, os.PathLike]) -> Tuple[np.ndarray, list, pd.DataFrame]:
    """Loads data from DeepLabCut (without using the package as a requirement)

    Parameters
    ----------
    dlcfile : Union[str, os.PathLike]
        Path to deeplabcut file

    Returns
    -------
    Tuple[np.ndarray, list, pd.DataFrame]
        Keypoint array
        List of names of bodyparts
        Pandas dataframe
    """
    assert os.path.isfile(dlcfile)
    # TODO: make function to load HDF5s
    ending = os.path.splitext(dlcfile)[1]
    assert ending == ".csv"

    # read the csv
    df = pd.read_csv(dlcfile, index_col=0)
    # 0th column is weird shit; set the columns to 0
    df.columns = df.iloc[1]
    # extract bodyparts; remove duplicates
    bodyparts = df.iloc[0].values[1:]
    bodyparts = bodyparts[::3].tolist()
    # get rid of weird rows
    df = df.iloc[2:, :]
    df = df.reset_index(drop=True)
    # get rid of scorer / bodyparts / coords column
    # df = df.iloc[:, 1:]

    # keypoints is of shape T x keypoints x 3
    keypoints = df.values.reshape(-1, len(bodyparts), 3).astype(np.float32)
    return keypoints, bodyparts, df


def stack_features_in_time(features: np.ndarray, frames_before_and_after: int = 15) -> np.ndarray:
    """For an array of keypoints, stack the frames before and after the current frame into one single vector.

    Parameters
    ----------
    features : np.ndarray
        T x K array of features
    frames_before_and_after : int, optional
        Number of frames before and after the current one to stack, by default 15

    Returns
    -------
    np.ndarray
        Stacked features
    """
    assert features.ndim == 2
    stacked_features = []
    N = features.shape[0]
    padded = np.pad(features, ((frames_before_and_after, frames_before_and_after), (0, 0)), mode="edge")

    for i in range(frames_before_and_after, N + frames_before_and_after):
        start_ind = i - frames_before_and_after
        end_ind = i + frames_before_and_after + 1

        stacked_features.append(padded[start_ind:end_ind, :].flatten())

    stacked = np.stack(stacked_features)
    assert stacked.shape[0] == features.shape[0]
    assert stacked.shape[1] == features.shape[1] * (frames_before_and_after * 2 + 1)
    return stacked


def expand_features_sturman(keypoints: np.ndarray, bodyparts: list, H: int, W: int) -> Tuple[np.ndarray, list]:
    """Expand 2D keypoints into features for behavioral classification.

    Based on Sturman et al. 2020:
        Sturman, O. et al. Deep learning-based behavioral analysis reaches human accuracy and is capable of
        outperforming commercial solutions. Neuropsychopharmacol. (2020) doi:10.1038/s41386-020-0776-y.

    Parameters
    ----------
    keypoints (np.ndarray)
        T x keypoints x 3 array. For this code, should be T x 7 x 3
    bodyparts (list)
        List of string names of bodyparts. For this code, assumes:
            ['nose', 'forepaw_l', 'forepaw_r', 'hindpaw_l', 'hindpaw_r', 'tailbase', 'tailtip', 'centroid']
    H (int)
        height of image. used for normalization
    W (int)
        width of image. used for normalization

    Returns
    -------
    features (np.ndarray)
        T x 44 array
    columns (list)
        44 element list with human-readable names identifying the features.
        will always be: ['nose_x', 'nose_y', 'forepaw_l_x', 'forepaw_l_y', 'forepaw_r_x',
        'forepaw_r_y', 'hindpaw_l_x', 'hindpaw_l_y', 'hindpaw_r_x', 'hindpaw_r_y', 'tailbase_x', 'tailbase_y',
        'tailtip_x', 'tailtip_y', 'centroid_x', 'centroid_y', 'nose_x_aligned', 'nose_y_aligned', 'forepaw_l_x_aligned',
        'forepaw_l_y_aligned', 'forepaw_r_x_aligned', 'forepaw_r_y_aligned', 'hindpaw_l_x_aligned',
        'hindpaw_l_y_aligned', 'hindpaw_r_x_aligned', 'hindpaw_r_y_aligned', 'tailbase_x_aligned', 'tailbase_y_aligned',
        'tailtip_x_aligned', 'tailtip_y_aligned', 'centroid_x_aligned', 'centroid_y_aligned', 'tail_angle',
        'forepaw_l_centroid_angle', 'forepaw_r_centroid_angle', 'hindpaw_l_centroid_angle', 'hindpaw_r_centroid_angle',
        'nose_tailbase_dist', 'tailbase_tailtip_dist', 'forepaw_hindpaw_dist', 'forepaw_nose_dist',
        'forepaw_forepaw_dist', 'hindpaw_hindpaw_dist', 'body_area']

        points up to "aligned" are in absolute coordinates, with top-left being (-1, -1) and bottom-right being (1, 1).
        "aligned" features are with tailbase at (0,0), rotated such that the nose is directly to the right.
        Distances are computed in normalized coordinates.

        All features are z-scored.
    """

    # add centroid as the 8th keypoint. mean of all paws
    keypoints = np.concatenate((keypoints, np.mean(keypoints[:, 1:5, :], axis=1, keepdims=True)), axis=1)
    bodyparts += ["centroid"]

    # normalize
    keypoints = normalize_keypoints(keypoints, H, W)

    from_tailbase_to_nose = keypoints[:, 0, :] - keypoints[:, 5, :]
    angles = np.arctan2(from_tailbase_to_nose[:, 1], from_tailbase_to_nose[:, 0])

    # https://en.wikipedia.org/wiki/Rotation_matrix
    rotmats = np.zeros((angles.shape[0], 2, 2), dtype=np.float32)
    # negative angles because we want to rotate everything to align to the body axis
    rotmats[:, 0, 0] = np.cos(-angles)
    rotmats[:, 0, 1] = -np.sin(-angles)
    rotmats[:, 1, 0] = np.sin(-angles)
    rotmats[:, 1, 1] = np.cos(-angles)

    origins = keypoints[:, 5, :]
    # subtract tailbase, rotate by negative body axis angle
    aligned = slow_alignment(keypoints, rotmats, origins)

    # nose-tailbase-tailtip angle
    tail_angle = np.arctan2(aligned[:, 6, 1], aligned[:, 6, 0])
    centroid_subtracted = aligned - aligned[:, 7:, :]
    # nose-centroid-paw angles
    paw_angles = np.arctan2(centroid_subtracted[:, 1:5, 1], centroid_subtracted[:, 1:5, 0])

    # l_forepaw, nose, r_forepaw, r_hindpaw, tailbase, l_hindpaw area. must be clockwise and in order!
    areas = np.array(
        [poly_area(aligned[i, [1, 0, 2, 4, 5, 3], 0], aligned[i, [1, 0, 2, 4, 5, 3], 1]) for i in range(len(aligned))]
    )

    nose_tailbase_distance = compute_distance(aligned[:, 0, :], aligned[:, 5, :])
    tailbase_tailtip_distance = compute_distance(aligned[:, 5, :], aligned[:, 6, :])
    forepaw_hindpaw_distance = (
        compute_distance(aligned[:, 1, :], aligned[:, 3, :]) + compute_distance(aligned[:, 2, :], aligned[:, 4, :])
    ) / 2
    forepaw_nose_distance = (
        compute_distance(aligned[:, 0, :], aligned[:, 1, :]) + compute_distance(aligned[:, 0, :], aligned[:, 2, :])
    ) / 2
    forepaw_forepaw_distance = compute_distance(aligned[:, 1, :], aligned[:, 2, :])
    hindpaw_hindpaw_distance = compute_distance(aligned[:, 3, :], aligned[:, 4, :])

    # go over our features, unpack if necessary, and add to one list
    features = []
    columns = []
    for i in range(len(bodyparts)):
        for j, coord in enumerate(["x", "y"]):
            features.append(keypoints[:, i, j])
            columns.append("{}_{}".format(bodyparts[i], coord))

    for i in range(len(bodyparts)):
        for j, coord in enumerate(["x", "y"]):
            features.append(aligned[:, i, j])
            columns.append("{}_{}_aligned".format(bodyparts[i], coord))

    features.append(tail_angle)
    columns.append("tail_angle")

    for i in range(4):
        features.append(paw_angles[:, i])
        columns.append("{}_centroid_angle".format(bodyparts[i + 1]))

    features.append(nose_tailbase_distance)
    columns.append("nose_tailbase_dist")
    features.append(tailbase_tailtip_distance)
    columns.append("tailbase_tailtip_dist")
    features.append(forepaw_hindpaw_distance)
    columns.append("forepaw_hindpaw_dist")
    features.append(forepaw_nose_distance)
    columns.append("forepaw_nose_dist")
    features.append(forepaw_forepaw_distance)
    columns.append("forepaw_forepaw_dist")
    features.append(hindpaw_hindpaw_distance)
    columns.append("hindpaw_hindpaw_dist")

    features.append(areas)
    columns.append("body_area")

    features = np.stack(features, axis=-1)
    # z-score
    denominator = features.std(axis=0, keepdims=True)
    denominator[denominator < 1e-6] = 1e-6
    z = (features - features.mean(axis=0, keepdims=True)) / denominator
    return z, columns
