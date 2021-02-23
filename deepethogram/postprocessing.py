import logging
from typing import Type, Tuple

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def remove_low_thresholds(thresholds: np.ndarray, minimum: float=0.01, f1s: np.ndarray=None, 
                          minimum_f1: float= 0.05) -> np.ndarray:
    """ Replaces thresholds below a certain value with 0.5
    
    If the model completely fails, the optimum threshold might be something erreoneous, such as 
    0.00001. This makes all predictions==1. 

    Parameters
    ----------
    thresholds : np.ndarray
        Shape (n_behaviors). Probabilities over this value will be set to 1
    minimum : float, optional
        Thresholds less than this value will be set to this value, by default 0.01
    f1s: np.ndarray, optional
        If submitted, f1s < value will also be set to 0. This again catches erroneous false-positives.
    Returns
    -------
    np.ndarray
        Thresholds with low values replaced
    """
    if np.sum(thresholds < minimum) > 0:
        indices = np.where(thresholds < minimum)[0]
        log.debug('thresholds {} too low, setting to {}'.format(thresholds[indices], minimum))
        thresholds[thresholds < minimum] = minimum
    if f1s is not None:
        if np.sum(f1s < minimum_f1) > 0:
            indices = np.where(f1s < minimum_f1)[0]
            log.debug('f1 {} too low, setting to 0.5'.format(f1s))
            thresholds[f1s < minimum_f1] = 0.5
    return thresholds

def get_onsets_offsets(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the onset and offset indices of a binary array.

    Onset: index at which the array goes from 0 -> 1 (the index with the 1, not the 0)
    offset: index at which the array goes from 1 -> 0 (the index with the 0, not the 1)

    Example:
        array = np.array([0, 0, 1, 1, 1, 0, 1])
        onsets, offsets = get_onsets_offsets(array)
        print(onsets, offsets) # [2 6] [5 7]

    Args:
        binary (np.ndarray): shape (N,) binary array

    Returns:
        onsets (np.ndarray): index at which a 0 switches to a 1
        offsets (np.ndarray): index at which a 1 switches to a 0
    """
    diff = np.diff(binary.astype(float))
    diff = np.concatenate([[0], diff])  # to align with data
    onsets = np.where(diff == 1)[0]
    if binary[0] == 1:
        onsets = np.concatenate(([0], onsets))
    offsets = np.where(diff == -1)[0]
    if binary[-1] == 1:
        offsets = np.concatenate((offsets, [len(binary)]))
    assert len(onsets) == len(offsets)
    return onsets, offsets


def get_bouts(ethogram: np.ndarray) -> list:
    """ Get bouts from an ethogram. Uses 1->0 and 0->1 changes to define bout starts and stops """
    K = ethogram.shape[1]
    stats = []
    for i in range(K):
        onsets, offsets = get_onsets_offsets(ethogram[:, i])
        stat = {'N': len(onsets),
                'lengths': np.array([offset - onset for (onset, offset) in zip(onsets, offsets)]),
                'starts': onsets,
                'ends': offsets}
        stats.append(stat)
    return stats


def find_bout_indices(predictions_trace: np.ndarray, bout_length: int, positive: bool = True) -> np.ndarray:
    """ Find indices where a bout of bout-length occurs in a binary vector

    Bouts are defined as consecutive sets of 1s (if `positive`) or 0s (if not `positive`).
    Parameters
    ----------
    predictions_trace: np.ndarray. shape (N, )
        binary vector corresponding to predictions for one class
    bout_length: int
        how many consecutive frames are in the bout
    positive: bool
        if True, find consecutive 1s with `bout_length`. Else, find consecutive 0s

    Returns
    -------
    expanded: np.ndarray
        indices in predictions_trace where a bout of length `bout_length` is occurring
    """
    # make a filter for convolution that will be 1 at that bout center
    center = np.ones(bout_length) / bout_length
    filt = np.concatenate([[-bout_length / 2], center, [-bout_length / 2]])
    if not positive:
        predictions_trace = np.logical_not(predictions_trace).astype(int)
    out = np.convolve(predictions_trace, filt, mode='same')
    indices = np.where(out == 1)[0]
    if len(indices) == 0:
        return np.array([]).astype(int)
    # if even, this corresponds to the center + 0.5 frame in the bout
    # if odd, this corresponds to the center frame of the bout
    # we want indices to contain the entire bout, not just the center frame
    if bout_length % 2:
        expanded = np.concatenate([np.array(range(i - bout_length // 2, i + bout_length // 2 + 1)) for i in indices])
    else:
        expanded = np.concatenate([np.array(range(i - bout_length // 2, i + bout_length // 2)) for i in indices])
    return expanded


def remove_short_bouts_from_trace(predictions_trace: np.ndarray, bout_length: int) -> np.ndarray:
    """ Removes bouts of length <= `bout_length` from a binary vector.

    Important note: we first remove "false negatives." e.g. if `bout_length` is 2, this will do something like:
        000111001111111011000010 ->  000111111111111111000010
    THEN we remove "false positives"
        000111111111111111000010 -> 000111111111111111000000

    Parameters
    ----------
    predictions_trace: np.ndarray. shape (N, )
        binary array of predictions
    bout_length: int
        consecutive sets of 0s or 1s of less than or equal to this length will be flipped.

    Returns
    -------
    predictions_trace: np.ndarray. shape (N, )
        binary predictions trace with short bouts removed
    """
    assert len(predictions_trace.shape) == 1, 'only 1D input: {}'.format(predictions_trace.shape)
    # first remove 1 frame bouts, then 2 frames, then 3 frames
    for bout_len in range(1, bout_length + 1):
        # first, remove "false negatives", like filling in gaps in true behavior bouts
        short_neg_indices = find_bout_indices(predictions_trace, bout_len, positive=False)
        predictions_trace[short_neg_indices] = 1
        # then remove "false positives", very short "1" bouts
        short_pos_indices = find_bout_indices(predictions_trace, bout_len)
        predictions_trace[short_pos_indices] = 0
    return predictions_trace


def remove_short_bouts(predictions: np.ndarray, bout_length: int) -> np.ndarray:
    """ Removes short bouts from a predictions array

    Applies `remove_short_bouts_from_trace` to each column of the input.

    Parameters
    ----------
    predictions: np.ndarray, shape (N, K)
        Array with N time points and K classes
    bout_length: int
        See remove_short_bouts_from_trace

    Returns
    -------
    predictions: np.ndarray, shape (N, K)
        Array of N timepoints and K classes with short bouts removed
    """
    assert len(predictions.shape) == 2, \
        '2D input to remove short bouts required (timepoints x classes): {}'.format(predictions.shape)

    T, K = predictions.shape
    for k in range(K):
        predictions[:, k] = remove_short_bouts_from_trace(predictions[:, k], bout_length)
    return predictions


def compute_background(predictions: np.ndarray) -> np.ndarray:
    """ Makes the background positive when no other behaviors are occurring

    Parameters
    ----------
    predictions: np.ndarray, shape (N, K)
        Binary predictions. The column `predictions[:, 0]` is the background class

    Returns
    -------
    predictions: np.ndarray, shape (N, K)
        Binary predictions. the background class is now the logical_not of whether or not there are any positive
        examples in the rest of the row
    """
    assert len(predictions.shape) == 2, 'predictions must be a TxK matrix: not {}'.format(predictions.shape)

    predictions[:, 0] = np.logical_not(np.any(predictions[:, 1:], axis=1)).astype(np.uint8)
    return predictions


class Postprocessor:
    """ Base class for postprocessing a set of input probabilities into predictions """
    def __init__(self, thresholds: np.ndarray, min_threshold=0.01):
        assert len(thresholds.shape) == 1, 'thresholds must be 1D array, not {}'.format(thresholds.shape)
        # edge case with poor thresholds, causes all predictions to be ==1
        thresholds = remove_low_thresholds(thresholds, minimum=min_threshold)
        self.thresholds = thresholds

    def threshold(self, probabilities: np.ndarray) -> np.ndarray:
        """ Applies thresholds to binarize inputs """
        assert len(probabilities.shape) == 2, 'probabilities must be a TxK matrix: not {}'.format(probabilities.shape)
        assert probabilities.shape[1] == self.thresholds.shape[0]
        predictions = (probabilities > self.thresholds).astype(int)
        return predictions

    def process(self, probabilities: np.ndarray) -> np.ndarray:
        """ Process probabilities. Will be overridden by subclasses """
        # the simplest form of postprocessing is just thresholding and making sure that background is the actual
        # logical_not of any other behavior. Therefore, its threshold is not used
        predictions = self.threshold(probabilities)
        predictions = compute_background(predictions)
        return predictions

    def __call__(self, probabilities: np.ndarray) -> np.ndarray:
        return self.process(probabilities)


class MinBoutLengthPostprocessor(Postprocessor):
    """ Postprocessor that removes bouts of length less than or equal to bout_length """
    def __init__(self, thresholds: np.ndarray, bout_length: int, **kwargs):
        super().__init__(thresholds, **kwargs)
        self.bout_length = bout_length

    def process(self, probabilities: np.ndarray) -> np.ndarray:
        predictions = self.threshold(probabilities)
        predictions = remove_short_bouts(predictions, self.bout_length)
        predictions = compute_background(predictions)
        return predictions


def get_postprocessor_from_cfg(cfg: DictConfig, thresholds: np.ndarray) -> Type[Postprocessor]:
    """ Returns a PostProcessor from an OmegaConf DictConfig returned by a  """
    if cfg.postprocessor.type is None:
        return Postprocessor(thresholds)
    elif cfg.postprocessor.type == 'min_bout':
        return MinBoutLengthPostprocessor(thresholds, cfg.postprocessor.min_bout_length)
    else:
        raise NotImplementedError
