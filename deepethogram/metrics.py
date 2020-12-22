import logging
import os
import warnings
from collections import defaultdict
from typing import Union, Tuple
from multiprocessing import Pool

import h5py
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score, auc

from deepethogram import utils

log = logging.getLogger(__name__)

# small epsilon to prevent divide by zero
EPS = 1e-7

# using multiprocessing on slurm causes a termination signal
try:
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm = True
except:
    slurm = False


def index_to_onehot(index: np.ndarray, n_classes: int) -> np.ndarray:
    """ Convert an array if indices to one-hot vectors.

    Parameters
    ----------
    index: np.ndarray. shape (N,)
        each element is the class of the correct label for that example
    n_classes: int
        Total number of classes. Necessary because this batch of indices might not have examples from all classes

    Returns
    -------
    onehot: shape (N, n_classes)
        Binary array with 1s

    Examples
    -------
        index_to_onehot(np.array([0, 1, 2, 3, 0]).astype(int), 4)
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [1, 0, 0, 0]], dtype=uint16)
    """
    onehot = np.zeros((index.shape[0], n_classes), dtype=np.uint16)
    onehot[np.arange(onehot.shape[0]), index] = 1
    return onehot


def hardmax(probabilities: np.ndarray) -> np.ndarray:
    """ Convert probability array to prediction by converting the max of each row to 1

    Parameters
    ----------
    probabilities: np.ndarray. Shape (N, K)
        probabilities output by some model. Floats between 0 and 1

    Returns
    -------
    array: np.ndarray. Shape (N, K)
        binary

    Examples
    -------
    # generate random array
    logits = np.random.uniform(size=(6,3))
    # dumb convert to probabilities
    probabilities = logits / logits.sum(axis=1)[:, np.newaxis]
    print(probabilities)
    array([[0.2600106 , 0.32258024, 0.41740916],
       [0.28634918, 0.4161426 , 0.29750822],
       [0.19937796, 0.32040531, 0.48021672],
       [0.70646227, 0.01531493, 0.2782228 ],
       [0.19636778, 0.35528756, 0.44834465],
       [0.78139017, 0.10704456, 0.11156526]])
    print(hardmax(probabilities))
    [[0 0 1]
     [0 1 0]
     [0 0 1]
     [1 0 0]
     [0 0 1]
     [1 0 0]]
    """
    # make an array of zeros
    array = np.zeros(probabilities.shape, dtype=np.uint16)
    # index into the array in the column with max probability,change it to 1
    array[np.arange(array.shape[0]), np.argmax(probabilities, axis=1)] = 1
    return array


def onehot_to_index(onehot: np.ndarray) -> np.ndarray:
    """Convert one-hot array to index by taking the argmax"""
    return np.argmax(onehot, axis=1)


def f1(predictions: np.ndarray, labels: np.ndarray, average: str = 'macro') -> np.ndarray:
    """ simple wrapper around sklearn.metrics.f1_score

    References
    -------
    [1]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    [2]: https://en.wikipedia.org/wiki/F1_score
    """
    # check to see if predictions are probabilities
    if predictions.dtype != np.int64 or predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    if labels.ndim > 1:
        labels = onehot_to_index(labels)
    F1 = f1_score(labels, predictions, average=average)
    return F1


def roc_auc(predictions: np.ndarray, labels: np.ndarray, average: str = 'macro') -> np.ndarray:
    """ simple wrapper around sklearn.metrics.roc_auc_score

    References
    -------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    .. [2] https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    if predictions.ndim == 1:
        raise ValueError('Predictions must be class probabilities before max!')
    if labels.ndim == 1:
        labels = index_to_onehot(labels, predictions.shape[1])
    score = roc_auc_score(labels, predictions, average=average)
    return score


def accuracy(predictions: np.ndarray, labels: np.ndarray):
    """ Return the fraction of elements in predictions that are equal to labels """
    return np.mean(predictions == labels)


def confusion(predictions: np.ndarray, labels: np.ndarray, K: int = None) -> np.ndarray:
    """ Computes confusion matrix. Much faster than sklearn.metrics.confusion_matrix for large numbers of predictions

    Parameters
    ----------
    predictions: np.ndarray. shape (N, ) or (N, K)
        can be probabilities, hardmax, or indicators
    labels: np.ndarray. shape (N,) or (N,K)
        can be one-hot or indicator
    K: int
        number of classes
    Returns
    -------
    cm: np.ndarray. shape (K, K)
        confusion matrix
    """
    if predictions.ndim > 1:
        K = predictions.shape[1]
        predictions = hardmax(predictions)  # prob -> onehot
        predictions = onehot_to_index(predictions)  # onehot -> index where 1
    if labels.ndim > 1:
        K = labels.shape[1]
        labels = onehot_to_index(labels)  # make sure labels are index
    if K is None:
        K = max(predictions.max() + 1, labels.max() + 1)

    cm = np.zeros((K, K)).astype(int)
    for i in range(K):
        for j in range(K):
            # these_inds = labels==i
            # cm[i, j] = np.sum((labels==i)*(predictions==j))
            cm[i, j] = np.sum(np.logical_and(labels == i, predictions == j))
    return cm


def binary_confusion_matrix(predictions, labels) -> np.ndarray:
    # behaviors x thresholds x 2 x 2
    # cms = np.zeros((K, N, 2, 2), dtype=int)
    ndim = predictions.ndim

    if ndim == 3:
        # 2 x 2 x K x N
        cms = np.zeros((2, 2, predictions.shape[1], predictions.shape[2]), dtype=int)
    elif ndim == 2:
        # 2 x 2 x K
        cms = np.zeros((2, 2, predictions.shape[1]), dtype=int)
    elif ndim == 1:
        # 2 x 2
        cms = np.zeros((2, 2), dtype=int)
    else:
        raise ValueError('unknown input shape: {}'.format(predictions.shape))

    neg_lab = np.logical_not(labels)
    neg_pred = np.logical_not(predictions)

    cms[0, 0] = (neg_lab * neg_pred).sum(axis=0)
    cms[0, 1] = (neg_lab * predictions).sum(axis=0)
    cms[1, 0] = (labels * neg_pred).sum(axis=0)
    cms[1, 1] = (labels * predictions).sum(axis=0)

    if ndim == 3:
        # output of shape 2 x 2 x N x K
        return cms.transpose(0, 1, 3, 2)
    # either 2 x 2 x K or just 2 x 2
    return cms


def binary_confusion_matrix_multiple_thresholds(probabilities, labels, thresholds):
    # this is the fastest I could possibly write it
    K = probabilities.shape[1]
    N = len(thresholds)

    pred = np.greater(probabilities.reshape(-1, 1), thresholds.reshape(1, -1)).reshape(-1, K, N)
    lab = labels.reshape(-1, 1).repeat(N, 1).reshape(-1, K, N)

    return binary_confusion_matrix(pred, lab)


def confusion_multiple_thresholds_alias(inp):
    # alias so that binary_confusion_matrix_multiple_thresholds only needs one tuple as input
    return binary_confusion_matrix_multiple_thresholds(*inp)


def confusion_alias(inp):
    return binary_confusion_matrix(*inp)


def binary_confusion_matrix_parallel(probs_or_preds, labels, thresholds=None, chunk_size: int = 100,
                                     num_workers: int = 4, parallel_chunk: int = 100):
    # log.info('num workers binary confusion parallel: {}'.format(num_workers))
    if slurm:
        parallel_chunk = 1
        num_workers = 1
    N = probs_or_preds.shape[0]

    starts = np.arange(0, N, chunk_size)
    ends = np.concatenate((starts[1:], [N]))

    if thresholds is not None:
        # probabilities
        iterator = ((probs_or_preds[start:end], labels[start:end], thresholds) for start, end in zip(starts, ends))
        cm = np.zeros((2, 2, len(thresholds), probs_or_preds.shape[1]), dtype=int)
        func = confusion_multiple_thresholds_alias
    else:
        # predictions
        iterator = ((probs_or_preds[start:end], labels[start:end]) for start, end in zip(starts, ends))
        if probs_or_preds.ndim == 2:
            cm = np.zeros((2, 2, probs_or_preds.shape[1]), dtype=int)
        elif probs_or_preds.ndim == 1:
            cm = np.zeros((2, 2), dtype=int)
        else:
            raise ValueError('weird shape in probs_or_preds: {}'.format(probs_or_preds.shape))
        func = confusion_alias
    # log.info('parallel start')
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for res in pool.imap_unordered(func, iterator, parallel_chunk):
                cm += res
    else:
        for args in iterator:
            cm += func(args)
    # log.info('parallel end')
    return cm


def compute_binary_confusion(predictions: np.ndarray, labels: np.ndarray,
                             thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ compute binary confusion matrices for input probabilities, labels, and thresholds. See confusion  """
    estimates = postprocess(predictions, thresholds, valid_bg=False)
    K = predictions.shape[1]

    cms = []
    for i in range(K):
        cm = confusion(estimates[:, i], labels[:, i], K=2)
        cms.append(cm)

    estimates = postprocess(predictions, thresholds, valid_bg=True)
    cms_valid_bg = []
    for i in range(K):
        cm = confusion(estimates[:, i], labels[:, i], K=2)
        cms_valid_bg.append(cm)
    return np.stack(cms), np.stack(cms_valid_bg)


def mean_class_accuracy(predictions, labels):
    """ computes the mean of diagonal elements of a confusion matrix """
    if predictions.ndim > 1:
        predictions = onehot_to_index(hardmax(predictions))
    if labels.ndim > 1:
        labels = onehot_to_index(labels)
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    on_diag = cm[np.where(np.eye(cm.shape[0], dtype=np.uint32))]
    return on_diag.mean()


def remove_invalid_values_predictions_and_labels(predictions: np.ndarray, labels: np.ndarray,
                                                 invalid_value: Union[int, float] = -1) -> \
        Tuple[np.ndarray, np.ndarray]:
    """ remove any rows where labels are equal to invalid_value.

    Used when (for example) the last sequence in a video is padded to have the proper sequence length. the padded inputs
    are paired with -1 labels, indicating that loss and metrics should not be applied there
    """
    is_invalid = labels == invalid_value
    valid_rows = np.logical_not(np.any(is_invalid, axis=1))
    predictions = predictions[valid_rows, :]
    labels = labels[valid_rows, :]
    return predictions, labels


def auc_on_array(x, y):
    if x.ndim < 2 or y.ndim < 2:
        return np.nan
    K = x.shape[1]
    assert K == y.shape[1]
    area_under_curve = np.zeros((K,), dtype=np.float32)
    for i in range(K):
        area_under_curve[i] = auc(x[:, i], y[:, i])
    return area_under_curve


def compute_metrics_by_threshold(probabilities, labels, thresholds, num_workers: int = 4, cm=None):
    # if we've computed cms elsewhere
    if cm is None:
        cm = binary_confusion_matrix_parallel(probabilities, labels, thresholds, num_workers=num_workers)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum(axis=0).sum(axis=0)
    p, r = compute_precision_recall(cm)
    tp, fp = compute_tpr_fpr(cm)
    info = compute_informedness(cm)
    f1 = compute_f1(p, r)
    fbeta_2 = compute_f1(p, r, beta=2.0)
    auroc = auc_on_array(fp, tp)
    mAP = auc_on_array(r, p)
    metrics_by_threshold = {
        'thresholds': thresholds,
        'accuracy': acc,
        'f1': f1,
        'precision': p,
        'recall': r,
        'fbeta_2': fbeta_2,
        'informedness': info,
        'tpr': tp,
        'fpr': fp,
        'auroc': auroc,
        'mAP': mAP,
        'confusion': cm
    }
    return metrics_by_threshold


def fast_auc(y_true, y_prob):
    if y_true.ndim == 2:
        return np.array([fast_auc(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])])
    # https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]

    n = len(y_true)

    nfalse = np.cumsum(1 - y_true)
    auc = np.cumsum((y_true * nfalse))[-1]
    # print(auc)
    auc /= (nfalse[-1] * (n - nfalse[-1]))
    return auc


# @profile
def evaluate_thresholds(probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray = None,
                        num_workers: int = 4) -> Tuple[dict, dict]:
    """ Given probabilities and labels, compute a bunch of metrics at each possible threshold value

    Also computes a number of metrics for which there is a single value for the input predictions / labels, something
    like the maximum F1 score across thresholds.

    Metrics computed for each threshold:
    Parameters
    ----------
    probabilities: np.ndarray. shape (N,K)
        output probabilities from some classifier
    labels: np.ndarray. shape (N, K) or (N,)
        binary or indicator labels. indicator labels will be converted to one-hot
    thresholds: np.ndarray. shape (M, )
        thresholds at which to convert probabilities into binary predictions.
        default value: np.linspace(0, 1, 101)

    Returns
    -------
    metrics_by_threshold: dict
        each value is an array of shape (M, ) or (M,K), with a value (or set of values) computed for each threshold
    epoch_metrics: dict
        each value is only a single float for the entire prediction / label set.
    """
    log.info('evaluating thresholds. P: {} lab: {} n_workers: {}'.format(probabilities.shape, labels.shape, num_workers))
    # log.info('SLURM in metrics file: {}'.format(slurm))
    if slurm and num_workers != 1:
        warnings.warn('using multiprocessing on slurm can cause issues. setting num_workers to 1')
        num_workers = 1

    if thresholds is None:
        # using 200 means that approximated mAP, AUROC is almost exactly the same as exact
        thresholds = np.linspace(0,1,200)
    # log.info('num workers in evaluate thresholds: {}'.format(num_workers))
    # log.debug('probabilities shape in metrics calc: {}'.format(probabilities.shape))
    metrics_by_threshold = {}
    if probabilities.ndim == 1:
        raise ValueError('To calc threshold, predictions must be probabilities, not classes')
    K = probabilities.shape[1]
    if labels.ndim == 1:
        labels = index_to_onehot(labels, K)

    probabilities, labels = remove_invalid_values_predictions_and_labels(probabilities, labels)
    # log.info('first metrics call')
    metrics_by_threshold = compute_metrics_by_threshold(probabilities, labels, thresholds, num_workers)
    # log.info('first metrics call finished')
    # log.info('finished computing binary confusion matrices')
    # optimum threshold: one that maximizes F1
    optimum_indices = np.argmax(metrics_by_threshold['f1'], axis=0)
    optimum_thresholds = thresholds[optimum_indices]

    # optimum info: maximizes informedness
    optimum_thresholds_info = thresholds[np.argmax(metrics_by_threshold['informedness'], axis=0)]

    metrics_by_threshold['optimum'] = optimum_thresholds
    metrics_by_threshold['optimum_info'] = optimum_thresholds_info

    # vectorized
    predictions = probabilities > optimum_thresholds
    # ALWAYS REPORT THE PERFORMANCE WITH "VALID" BACKGROUND
    predictions[:, 0] = np.logical_not(np.any(predictions[:, 1:], axis=1))
    
    # log.info('computing metric thresholds again')
    # re-use our confusion matrix calculation. returns N x N x K values
    # log.info('second metircs call')
    metrics_by_class = compute_metrics_by_threshold(predictions, labels, None, num_workers)
    # log.info('second metrics call ended')

    # summing over classes is the same as flattening the array. ugly syntax
    # TODO: make function that computes metrics from a stack of confusion matrices rather than this none None business
    # log.info('third metrics call')
    overall_metrics = compute_metrics_by_threshold(None, None, thresholds=None, num_workers=num_workers,
                                                   cm=metrics_by_class['confusion'].sum(axis=2))
    # log.info('third metrics call ended')
    epoch_metrics = {
        'accuracy_overall': overall_metrics['accuracy'],
        'accuracy_by_class': metrics_by_class['accuracy'],
        'f1_overall': overall_metrics['f1'],
        'f1_class_mean': metrics_by_class['f1'].mean(),
        'f1_by_class': metrics_by_class['f1'],
        'binary_confusion': metrics_by_class['confusion'].transpose(2, 0, 1),
        'auroc_by_class': metrics_by_threshold['auroc'],
        'auroc_class_mean': metrics_by_threshold['auroc'].mean(),
        'mAP_by_class': metrics_by_threshold['mAP'],
        'mAP_class_mean': metrics_by_threshold['mAP'].mean(),
        # to compute these, would need to make confusion matrices on flattened array, which is slow
        'auroc_overall': np.nan,
        'mAP_overall': np.nan
    }
    # it is too much of a pain to increase the speed on roc_auc_score and mAP
    # try:
    #     epoch_metrics['auroc_overall'] = roc_auc_score(labels, probabilities, average='micro')
    #     epoch_metrics['auroc_by_class'] = roc_auc_score(labels, probabilities, average=None)
    #     # small perf improvement is not worth worrying about bugs
    #     # epoch_metrics['auroc_overall'] = fast_auc(labels.flatten(), probabilities.flatten())
    #     # epoch_metrics['auroc_by_class'] = fast_auc(labels, probabilities)
    #     epoch_metrics['auroc_class_mean'] = epoch_metrics['auroc_by_class'].mean()
    # except ValueError:
    #     # only one class in labels...
    #     epoch_metrics['auroc_overall'] = np.nan
    #     epoch_metrics['auroc_class_mean'] = np.nan
    #     epoch_metrics['auroc_by_class'] = np.array([np.nan for _ in range(K)])
    #
    # epoch_metrics['mAP_overall'] = average_precision_score(labels, probabilities, average='micro')
    # epoch_metrics['mAP_by_class'] = average_precision_score(labels, probabilities, average=None)
    # # this is a misnomer: mAP by class is just AP
    # epoch_metrics['mAP_class_mean'] = epoch_metrics['mAP_by_class'].mean()
    # log.info('returning metrics')
    return metrics_by_threshold, epoch_metrics


def compute_tpr_fpr(cm: np.ndarray) -> Tuple[float, float]:
    """ compute true positives and false positives from a non-normalized confusion matrix """
    # normalize so that each are rates
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    fp = cm_normalized[0, 1]
    tp = cm_normalized[1, 1]
    return tp, fp


def get_denominator(expression: Union[float, np.ndarray]):
    if isinstance(expression, (int, np.integer, float, np.floating)):
        return max(EPS, expression)
    # it's an array
    # convert to floating point type-- if it's integer, it will just ignore the eps and not throw an error
    expression = expression.astype(np.float32)
    expression[expression < EPS] = EPS
    return expression


def compute_f1(precision: float, recall: float, beta: float = 1.0) -> float:
    """ compute f1 if you already have precison and recall. Prevents re-computing confusion matrix, etc """

    num = (1 + beta ** 2) * (precision * recall)
    denom = get_denominator((beta ** 2) * precision + recall)
    return num / denom


def compute_precision_recall(cm: np.ndarray) -> Tuple[float, float]:
    """ computes precision and recall from a confusion matrix """
    tn = cm[0, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    precision = tp / get_denominator(tp + fp)
    recall = tp / get_denominator(tp + fn)
    return precision, recall


def compute_mean_accuracy(cm: np.ndarray) -> float:
    """ compute the mean of true positive rate and true negative rate from a confusion matrix """
    cm = cm.astype('float') / get_denominator(cm.sum(axis=1)[:, np.newaxis])
    tp = cm[1, 1]
    tn = cm[0, 0]
    return np.mean([tp, tn])


def compute_informedness(cm: np.ndarray, eps: float = 1e-7) -> float:
    """ compute informedness from a confusion matrix. Also known as Youden's J statistic

    Parameters
    ----------
    cm: np.ndarray
        confusion matrix
    eps: float
        small value to prevent divide by zero

    Returns
    -------
    informedness: float
        Ranges from 0 to 1. Gives equal weight to false positives and false negatives.

    References
    -------
    .. [1]: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    """
    tn = cm[0, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    sensitivity = tp / get_denominator(tp + fn)
    specificity = tn / get_denominator(fp + tn)
    return sensitivity + specificity - 1


def postprocess(predictions: np.ndarray, thresholds: np.ndarray, valid_bg: bool = True) -> np.ndarray:
    """ Turn probabilities into predictions, with special handling of background.
    TODO: Should be removed in favor of deepethogram.prostprocessing
    """
    N, n_classes = predictions.shape
    assert (len(thresholds) == n_classes)
    estimates = np.zeros((N, n_classes), dtype=np.int64)

    for i in range(0, n_classes):
        estimates[:, i] = (predictions[:, i] > thresholds[i]).astype(int)
    if valid_bg:
        estimates[:, 0] = np.logical_not(np.any(estimates[:, 1:], axis=1)).astype(int)
    return estimates


all_metrics = {
    'accuracy': accuracy,
    'mean_class_accuracy': mean_class_accuracy,
    'f1': f1,
    'roc_auc': roc_auc,
    'confusion': binary_confusion_matrix
}


def list_to_mean(values):
    if type(values[0]) == torch.Tensor:
        value = utils.tensor_to_np(torch.stack(values).mean())
    elif type(values[0]) == np.ndarray:
        if values[0].size == 1:
            value = np.stack(np.array(values)).mean()
        else:
            value = np.concatenate(np.array(values)).mean()
    else:
        raise TypeError('Input should be numpy array or torch tensor. Type: ', type(values[0]))
    return value


def append_to_hdf5(f, name, value, axis=0):
    """ resizes an HDF5 dataset and appends value """
    f[name].resize(f[name].shape[axis] + 1, axis=axis)
    f[name][-1] = value


class Buffer:
    def __init__(self):
        self.data = {}
        self.splits = ['train', 'val', 'test', 'speedtest']
        for split in self.splits:
            self.initialize(split)

    def initialize(self, split):
        self.data[split] = defaultdict(list)

    def append(self, split: str, data: dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # don't convert to numpy for speed
                value = value.detach().cpu()
            self.data[split][key].append(value)

    def stack(self, split):
        stacked = {}
        keys = list(self.data[split].keys())
        # go by key so we can delete each value from memory after stacking
        for key in keys:
            value = self.data[split][key]
            first_element = value[0]
            if isinstance(value, (int, float, np.integer, np.floating, np.ndarray, list)):
                try:
                    # default is concatenating along the batch dimension
                    value = np.concatenate(value)
                except ValueError:
                    # input is likely just a list
                    value = np.stack(value)
            elif isinstance(first_element, torch.Tensor):
                value = torch.stack(value)
            stacked[key] = value
            del self.data[split][key]

        self.initialize(split)
        return stacked

    def clear(self, split=None):
        if split is None:
            for split in self.data.keys():
                self.clear(split)
        keys = list(self.data[split].keys())
        for key in keys:
            del self.data[split][key]
        self.data[split] = defaultdict(list)


class EmptyBuffer:
    def __init__(self):
        self.data = {}
        self.splits = ['train', 'val', 'test', 'speedtest']
        for split in self.splits:
            self.initialize(split)

    def initialize(self, split):
        self.data[split] = defaultdict(list)

    def append(self, split: str, data: dict):
        pass

    def stack(self, split):
        pass

    def clear(self, split=None):
        pass


class Metrics:
    """Class for saving a list of per-epoch metrics to disk as an HDF5 file"""

    def __init__(self, run_dir: Union[str, bytes, os.PathLike],
                 metrics: list,
                 key_metric: str,
                 name: str,
                 num_parameters: int,
                 splits: list = ['train', 'val'],
                 num_workers: int = 4):
        """ Metrics constructor

        Parameters
        ----------
        run_dir: str, os.PathLike
            directory into which to save metrics file
        metrics: list
            list of metrics. a dataset will be created in the HDF5 file for each of these, for each split
        key_metric: str
            which metric is considered the "key". This can be used for determining when a model has converged, etc.
        name: str
            filename will be /run_dir/{name}_metrics.h5
        num_parameters: int
            number of parameters in your model. useful to save this for later
        splits: list
            either ['train', 'val'] or ['train', 'val', 'test']
        """
        assert (os.path.isdir(run_dir))
        assert key_metric in metrics or key_metric == 'loss'
        self.fname = os.path.join(run_dir, '{}_metrics.h5'.format(name))
        log.debug('making metrics file at {}'.format(self.fname))

        self.metrics = metrics
        self.key_metric = key_metric
        self.splits = splits
        self.num_parameters = num_parameters
        self.learning_rate = None
        self.initialize_file()
        self.num_workers = num_workers

        self.buffer = Buffer()
        self.latest_key = {}
        self.latest_loss = {}

    def update_lr(self, lr):
        self.learning_rate = lr

    def compute(self, data: dict) -> dict:
        """ Computes metrics from one epoch's batch of data

        Args:
            data: dict
                dict of Numpy arrays containing any data needed to compute metrics

        Returns:
            metrics: dict
                dict of numpy arrays / floats containing metrics to be written to disk
        """
        metrics = {}
        keys = list(data.keys())
        if 'loss' in keys:
            metrics['loss'] = np.mean(data['loss'])
        if 'time' in keys:
            # assume it's seconds per image
            FPS = 1 / get_denominator(np.mean(data['time']))
            metrics['fps'] = FPS
        elif 'fps' in keys:
            FPS = np.mean(data['fps'])
            metrics['fps'] = FPS
        if 'lr' in keys:
            # note: this should always be a scalar, but set to mean just in case there's multiple
            metrics['lr'] = np.mean(data['lr'])
        return metrics

    def initialize_file(self):
        mode = 'r+' if os.path.isfile(self.fname) else 'w'
        with h5py.File(self.fname, mode) as f:
            f.attrs['num_parameters'] = self.num_parameters
            f.attrs['key_metric'] = self.key_metric
            # make an HDF5 group for each split
            for split in self.splits:
                group = f.create_group(split)
                # all splits and datasets will have loss values-- others will come from self.compute()
                group.create_dataset('loss', (0,), maxshape=(None,), dtype=np.float32)

    def save_metrics_to_disk(self, metrics: dict, split: str) -> None:
        with h5py.File(self.fname, 'r+') as f:
            # utils.print_hdf5(f)
            if split not in f.keys():
                # should've created top-level groups in initialize_file; this is for nesting
                f.create_group(split)
            group = f[split]
            datasets = list(group.keys())
            for key, array in metrics.items():
                if isinstance(array, (int, float, np.integer, np.floating)):
                    array = np.array(array)
                # ALLOW FOR NESTING
                if isinstance(array, dict):
                    group_name = split + '/' + key
                    self.save_metrics_to_disk(array, group_name)
                elif isinstance(array, np.ndarray):
                    if key in datasets:
                        # expand along the epoch dimension
                        group[key].resize(group[key].shape[0] + 1, axis=0)
                    else:
                        # create dataset
                        shape = (1, *array.shape)
                        maxshape = (None, *array.shape)
                        log.debug('creating dataset {}/{}: shape {}'.format(split, key, shape))
                        group.create_dataset(key, shape, maxshape=maxshape, dtype=array.dtype)
                    group[key][-1] = array
                else:
                    raise ValueError('Metrics must contain dicts of np.ndarrays, not {} of type {}'.format(array,
                                                                                                           type(array)))

    def end_epoch(self, split: str):
        """ End the current training epoch. Saves any metrics in memory to disk

        Parameters
        ----------
        split: str
            which epoch just ended. train, validation, test, and speedtest are treated differently
        """
        data = self.buffer.stack(split)
        metrics = self.compute(data)

        # import pdb; pdb.set_trace()

        if split != 'speedtest':
            assert 'loss' in data.keys()

            # store most recent loss and key metric as attributes, for use in scheduling, stopping, etc.
            self.latest_loss[split] = metrics['loss']
            self.latest_key[split] = metrics[self.key_metric]

        self.save_metrics_to_disk(metrics, split)

    def __getitem__(self, inp: tuple) -> np.ndarray:
        split, metric_name, epoch_number = inp
        with h5py.File(self.fname, 'r') as f:
            assert split in f.keys(), 'split {} not found in file: {}'.format(split, list(f.keys()))
            group = f[split]
            assert metric_name in group.keys(), 'metric {} not found in group: {}'.format(metric_name,
                                                                                          list(group.keys()))
            data = group[metric_name][epoch_number, ...]
        return data


class EmptyMetrics(Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(os.getcwd(), [], 'loss', 'empty', 0)
        self.buffer = EmptyBuffer()
        self.key_metric = 'loss'

    def end_epoch(self, split, *args, **kwargs):
        # calling this clears the buffer
        self.buffer.clear(split)

    def initialize_file(self):
        pass


class Classification(Metrics):
    """ Metrics class for saving multiclass or multilabel classifcation metrics to disk """

    def __init__(self, run_dir: Union[str, bytes, os.PathLike], key_metric: str, num_parameters: int,
                 num_classes: int = None, metrics: list = ['accuracy', 'mean_class_accuracy', 'f1', 'roc_auc'],
                 splits: list = ['train', 'val'],
                 ignore_index: int = -1, evaluate_threshold: bool = False, num_workers: int = 4):
        """ Constructor for classification metrics class

        Parameters
        ----------
        run_dir
            see Metrics
        key_metric
            see Metrics
        num_parameters
            see Metrics
        num_classes: int
            number of classes (behaviors) in your classification problem
        metrics: list
            each string in this list corresponds to a function that operates on probabilities and labels
        splits: list
            see Metrics
        ignore_index: int
            labels with this index will be masked for the purposes of computing metrics
        evaluate_threshold: bool
            Hack for multi-label classification problems. If True, at each epoch will compute a bunch of metrics for
            each potential threshold. See evaluate_thresholds
        """
        super().__init__(run_dir, metrics, key_metric, 'classification', num_parameters, splits, num_workers)

        self.metric_funcs = all_metrics

        if 'confusion' in metrics:
            assert (num_classes is not None)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.evaluate_threshold = evaluate_threshold

        # if self.evaluate_threshold:
        #     self.thresholds = np.linspace(0, 1, 101)

    def stack_sequence_data(self, array: np.ndarray) -> np.ndarray:
        # if probs or labels are one-hot N x K or indicator N, return
        if array.ndim < 3:
            return array
        assert array.ndim == 3
        if array.shape[1] < array.shape[2]:
            N, K, T = array.shape
            array = array.transpose(0, 2, 1).reshape(N * T, K)
        else:
            N, T, K = array.shape
            array = array.reshape(N * T, K)
        return array

    def compute(self, data: dict):
        # computes mean loss, etc
        metrics = super().compute(data)

        if 'probs' not in data.keys():
            # might happen during speedtest
            return metrics

        # if data are from sequence models, stack into N*T x K not N x K x T
        probs = self.stack_sequence_data(data['probs'])
        labels = self.stack_sequence_data(data['labels'])

        num_classes = probs.shape[1]
        one_hot = probs.shape[-1] == labels.shape[-1]
        if one_hot:
            rows_with_false_labels = np.any(labels == self.ignore_index, axis=1)
        else:
            rows_with_false_labels = labels == self.ignore_index

        true_rows = np.logical_not(rows_with_false_labels)
        probs = probs[true_rows, :]
        labels = labels[true_rows, :] if one_hot else labels[true_rows]

        if self.evaluate_threshold:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics_by_threshold, epoch_metrics = evaluate_thresholds(probs, labels, None,
                                                                          self.num_workers)
                metrics['metrics_by_threshold'] = metrics_by_threshold
                for key, value in epoch_metrics.items():
                    metrics[key] = value
        else:
            # multiclass classification, not multilabel
            if one_hot:
                labels = onehot_to_index(labels)

            predictions = np.argmax(probs, axis=1)

            with warnings.catch_warnings():
                for metric in self.metrics:
                    if metric == 'confusion':
                        warnings.simplefilter("ignore")
                        metrics[metric] = confusion(predictions, labels, K=self.num_classes)
                        # import pdb
                        # pdb.set_trace()
                    elif metric == 'binary_confusion':
                        pass
                    else:
                        warnings.simplefilter("ignore")
                        metrics[metric] = self.metric_funcs[metric](predictions, labels)
        return metrics


class OpticalFlow(Metrics):
    """ Metrics class for saving optic flow metrics to disk """

    def __init__(self, run_dir, key_metric, num_parameters, metrics=['SSIM_full'],
                 splits=['train', 'val']):
        super().__init__(run_dir, metrics, key_metric, 'opticalflow', num_parameters, splits)

    def compute(self, data: dict) -> dict:
        """ Computes metrics from one epoch's batch of data

        Args:
            data: dict
                dict of Numpy arrays containing any data needed to compute metrics

        Returns:
            metrics: dict
                dict of numpy arrays / floats containing metrics to be written to disk
        """
        metrics = super().compute(data)

        for key in ['SSIM', 'L1', 'smoothness', 'sparsity', 'L1']:
            if key in data.keys():
                metrics[key] = data[key].mean()
        return metrics
