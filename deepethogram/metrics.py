import logging
import os
import warnings
from collections import defaultdict
from typing import Union, Tuple

import h5py
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score

from deepethogram import utils

log = logging.getLogger(__name__)


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
            cm[i, j] = np.sum(np.logical_and(labels == i, predictions == j))
            # cm[i,j] = np.sum(predictions[these_inds]==j)
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


def evaluate_thresholds(predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> Tuple[dict, dict]:
    """ Given probabilities and labels, compute a bunch of metrics at each possible threshold value

    Also computes a number of metrics for which there is a single value for the input predictions / labels, something
    like the maximum F1 score across thresholds.

    Metrics computed for each threshold:
    Parameters
    ----------
    predictions: np.ndarray. shape (N,K)
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
    metrics_by_threshold = {}
    if predictions.ndim == 1:
        raise ValueError('To calc threshold, predictions must be probabilities, not classes')
    K = predictions.shape[1]
    if labels.ndim == 1:
        labels = index_to_onehot(labels, K)

    predictions, labels = remove_invalid_values_predictions_and_labels(predictions, labels)
    # TODO: refactor this to use a defaultdict(list)
    accuracy_by_class = []
    f1_by_class = []
    precision_by_class = []
    recall_by_class = []
    mean_acc_by_class = []
    informedness_by_class = []
    tpr_by_class = []
    fpr_by_class = []
    log.debug('Evaluating multiple metrics for many thresholds')
    for i, thresh in enumerate(thresholds):
        estimated = (predictions > thresh).astype(int)
        accuracy_by_class.append(np.mean(estimated == labels, axis=0))
        # f1_by_class.append( [f1_score(labels[:,j], estimated[:,j]) for j in range(K)] )
        precision, recall = [], []
        mean_acc = []
        informedness = []
        f1_val = []
        tpr, fpr = [], []
        for j in range(K):
            cm = confusion(estimated[:, j], labels[:, j], K=2)
            # import pdb; pdb.set_trace()
            p, r = compute_precision_recall(cm)
            tp, fp = compute_tpr_fpr(cm)
            ma = compute_mean_accuracy(cm)
            info = compute_informedness(cm)
            f1_val.append(compute_f1(p, r))
            mean_acc.append(ma)
            precision.append(p)
            recall.append(r)
            informedness.append(info)
            tpr.append(tp)
            fpr.append(fp)

        precision = np.stack(precision)
        recall = np.stack(recall)
        mean_acc = np.stack(mean_acc)
        informedness = np.stack(informedness)
        f1_by_class.append(f1_val)
        precision_by_class.append(precision)
        recall_by_class.append(recall)
        mean_acc_by_class.append(mean_acc)
        informedness_by_class.append(informedness)
        tpr_by_class.append(np.stack(tpr))
        fpr_by_class.append(np.stack(fpr))
    accuracy_by_class = np.stack(accuracy_by_class, axis=0)
    f1_by_class = np.stack(f1_by_class, axis=0)
    precision_by_class = np.stack(precision_by_class, axis=0)
    recall_by_class = np.stack(recall_by_class, axis=0)
    mean_acc_by_class = np.stack(mean_acc_by_class, axis=0)
    informedness_by_class = np.stack(informedness_by_class, axis=0)
    tpr_by_class = np.stack(tpr_by_class, axis=0)
    fpr_by_class = np.stack(fpr_by_class, axis=0)

    metrics_by_threshold = {'thresholds': thresholds, 'accuracy': accuracy_by_class, 'f1': f1_by_class,
                            'precision': precision_by_class, 'recall': recall_by_class,
                            'mean_accuracy': mean_acc_by_class, 'informedness': informedness_by_class,
                            'tpr': tpr_by_class, 'fpr': fpr_by_class}

    # optimum threshold: one that maximizes F1
    optimum_thresholds = np.zeros((K,), dtype=np.float32)
    for i in range(K):
        # ax.plot(t, f1[:,i], label=class_names[i])
        index = np.argmax(f1_by_class[:, i])
        max_acc = f1_by_class[index, i]
        optimum_thresholds[i] = thresholds[index]

    # optimum info: maximizes informedness
    optimum_thresholds_info = np.zeros((K,), dtype=np.float32)
    for i in range(K):
        index = np.argmax(informedness_by_class[:, i])
        max_acc = informedness_by_class[index, i]
        optimum_thresholds_info[i] = thresholds[index]

    metrics_by_threshold['optimum'] = optimum_thresholds
    metrics_by_threshold['optimum_info'] = optimum_thresholds_info

    heuristic_predictions = np.zeros_like(labels)
    independent_predictions = np.zeros_like(labels)
    for i in range(0, K):
        heuristic_predictions[:, i] = (predictions[:, i] > optimum_thresholds[i]).astype(int)
        independent_predictions[:, i] = (predictions[:, i] > optimum_thresholds[i]).astype(int)
    heuristic_predictions[:, 0] = np.logical_not(np.any(heuristic_predictions[:, 1:], axis=1)).astype(int)

    # metrics_by_threshold[]

    epoch_metrics = {'accuracy': np.mean(independent_predictions == labels),
                     'accuracy_valid_bg': np.mean(heuristic_predictions == labels),
                     'f1_overall': f1_score(labels, independent_predictions, average='micro'),
                     'f1_by_class': f1_score(labels, independent_predictions, average='macro'),
                     'f1_overall_valid_bg': f1_score(labels, heuristic_predictions, average='micro'),
                     'f1_by_class_valid_bg': f1_score(labels, heuristic_predictions, average='macro')}
    try:
        epoch_metrics['auroc'] = roc_auc_score(labels, predictions, average='micro')
    except ValueError:
        # print('only one class in labels...')
        epoch_metrics['auroc'] = np.nan
    try:
        epoch_metrics['auroc_by_class'] = roc_auc_score(labels, predictions, average='macro')
    except ValueError:
        # print('only one class in labels...')
        epoch_metrics['auroc_by_class'] = np.nan

    epoch_metrics['mAP'] = average_precision_score(labels, predictions, average='micro')
    epoch_metrics['mAP_by_class'] = average_precision_score(labels, predictions, average='macro')

    cms, cms_valid_bg = compute_binary_confusion(predictions, labels, optimum_thresholds)
    epoch_metrics['binary_confusion'] = cms
    epoch_metrics['binary_confusion_valid'] = cms_valid_bg

    return metrics_by_threshold, epoch_metrics


def compute_tpr_fpr(cm: np.ndarray) -> Tuple[float, float]:
    """ compute true positives and false positives from a non-normalized confusion matrix """
    # normalize so that each are rates
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    fp = cm_normalized[0, 1]
    tp = cm_normalized[1, 1]
    return tp, fp


def compute_f1(precision: float, recall: float) -> float:
    """ compute f1 if you already have precison and recall. Prevents re-computing confusion matrix, etc """
    return 2 * (precision * recall) / (precision + recall + 1e-7)


def compute_precision_recall(cm: np.ndarray) -> Tuple[float, float]:
    """ computes precision and recall from a confusion matrix """
    tn = cm[0, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return precision, recall


def compute_mean_accuracy(cm: np.ndarray) -> float:
    """ compute the mean of true positive rate and true negative rate from a confusion matrix """
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
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

    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (fp + tn + eps)
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
    'confusion': confusion
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


class Metrics:
    """Class for saving a list of per-epoch metrics to disk as an HDF5 file"""

    def __init__(self, run_dir: Union[str, bytes, os.PathLike],
                 metrics: list,
                 key_metric: str,
                 name: str,
                 num_parameters: int,
                 splits: list = ['train', 'val']):
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

        self.learning_rate = None
        mode = 'r+' if os.path.isfile(self.fname) else 'w'
        with h5py.File(self.fname, mode) as f:
            f.attrs['num_parameters'] = num_parameters
            f.attrs['key_metric'] = key_metric
            f.create_dataset('learning_rates', (0,), maxshape=(None,), dtype=np.float64)
            # make an HDF5 group for each split
            for split in splits:
                f.create_group(split)
                # save loss and time for each split
                f.create_dataset(split + '/' + 'time', (0,), maxshape=(None,), dtype=np.float64)
                f.create_dataset(split + '/' + 'loss', (0,), maxshape=(None,), dtype=np.float64)

                # create a dataset in this group
                for metric in metrics:
                    dset_name = split + '/' + metric
                    # hack
                    if metric == 'confusion':
                        # this metric has a different shape than the rest, so we'll add this to
                        continue
                    else:
                        shape = (0,)
                        maxshape = (None,)
                    # print(metric, shape, maxshape)
                    f.create_dataset(dset_name, shape, maxshape=maxshape, dtype=np.float64)

            f.create_dataset('test' + '/' + 'time', (0,), maxshape=(None,), dtype=np.float64)

        self.epoch_predictions = []
        self.epoch_labels = []
        self.epoch_t = []
        self.epoch_loss = []
        self.splits = splits
        self.loss_components = defaultdict(list)
        self.latest_key = {}
        self.latest_loss = {}

    def loss_append(self, loss):
        self.epoch_loss.append(loss)

    def time_append(self, t):
        self.epoch_t.append(t)

    def update_lr(self, lr):
        self.learning_rate = lr

    def end_epoch_speedtest(self):
        times = np.array(self.epoch_t)
        mean_time = times.mean()
        with h5py.File(self.fname, 'r+') as f:
            append_to_hdf5(f, 'test' + '/' + 'time', mean_time)
        self.epoch_t = []

    def loss_components_append(self, loss_dict):
        for metric, value in loss_dict.items():
            if type(value) == torch.Tensor:
                value = utils.tensor_to_np(value)
            self.loss_components[metric].append(value)

    def end_epoch(self, split: str):
        """ End the current training epoch. Saves any metrics in memory to didk

        Parameters
        ----------
        split: str
            which epoch just ended. train, validation, test, and speedtest are treated differently
        """
        self.latest_key = {}

        times = np.array(self.epoch_t)
        mean_time = times.mean()
        mean_loss = np.array(self.epoch_loss).mean()
        self.latest_loss = {}
        self.latest_loss[split] = mean_loss

        # Save information in the "loss components" dictionary to disk
        with h5py.File(self.fname, 'r+') as f:
            for metric, value in self.loss_components.items():
                name = split + '/' + metric
                value = list_to_mean(value)

                append_to_hdf5(f, name, value)
                if metric == self.key_metric:
                    self.latest_key[split] = value

            append_to_hdf5(f, split + '/' + 'time', mean_time)
            append_to_hdf5(f, split + '/' + 'loss', mean_loss)
            append_to_hdf5(f, 'learning_rates', self.learning_rate)
        if self.key_metric == 'loss':
            self.latest_key[split] = self.latest_loss[split]
        if len(self.epoch_predictions) != 0 and len(self.epoch_labels) != 0:
            # some subclasses, instead of storing metrics as loss components, will store a set of predictions and labels
            # we want to compute metrics all at once instead of on a per-batch basis
            self.compute_metrics_from_batches_labels(split)

        self.epoch_t = []
        self.epoch_loss = []
        self.epoch_predictions = []
        self.epoch_labels = []
        self.loss_components = defaultdict(list)

    def compute_metrics_from_batches_labels(self, split):
        # to be overwritten by subclass
        raise NotImplementedError


class Classification(Metrics):
    """ Metrics class for saving multiclass or multilabel classifcation metrics to disk """

    def __init__(self, run_dir: Union[str, bytes, os.PathLike], key_metric: str, num_parameters: int,
                 num_classes: int = None, metrics: list = ['accuracy', 'mean_class_accuracy', 'f1', 'roc_auc'],
                 splits: list = ['train', 'val'],
                 ignore_index: int = -1, evaluate_threshold: bool = False):
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
        super().__init__(run_dir, metrics, key_metric, 'classification', num_parameters, splits)

        self.metric_funcs = all_metrics

        if 'confusion' in metrics:
            assert (num_classes is not None)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.evaluate_threshold = evaluate_threshold

        # custom initialization because confusion matrices don't have usual shape
        with h5py.File(self.fname, 'r+') as f:
            for split in self.splits:
                # print('{} keys: {}'.format(split, list(f[split].keys())))
                for metric in metrics:
                    if metric == 'confusion':
                        dset_name = split + '/' + metric
                        shape = (0, num_classes, num_classes)
                        maxshape = (None, num_classes, num_classes)
                        # print(metric, shape, maxshape)
                        f.create_dataset(dset_name, shape, maxshape=maxshape, dtype=np.float64)

        if self.evaluate_threshold:
            self.thresholds = np.linspace(0, 1, 101)
            self.threshold_curves = ['thresholds', 'accuracy', 'f1', 'optimum', 'precision', 'recall',
                                     'mean_accuracy', 'informedness', 'optimum_info', 'tpr', 'fpr']
            self.threshold_metrics = ['accuracy', 'accuracy_valid_bg', 'f1_overall', 'f1_by_class',
                                      'f1_overall_valid_bg', 'f1_by_class_valid_bg',
                                      'auroc', 'auroc_by_class',
                                      'mAP', 'mAP_by_class',
                                      'binary_confusion', 'binary_confusion_valid']
            self.metrics_by_threshold = {}

        with h5py.File(self.fname, 'r+') as f:
            if self.evaluate_threshold:
                g = f.create_group('thresholds')
            for split in splits:
                if self.evaluate_threshold:
                    g2 = g.create_group(split)
                    for metric in self.threshold_metrics:
                        if 'binary_confusion' in metric:
                            shape = (0, num_classes, 2, 2)
                            maxshape = (None, num_classes, 2, 2)
                        else:
                            shape = (0,)
                            maxshape = (None,)
                        g2.create_dataset(metric, shape, maxshape=maxshape, dtype=np.float64)

    def batch_append(self, predictions, labels):
        if type(predictions) == torch.Tensor:
            predictions = utils.tensor_to_np(predictions)
        if type(labels) == torch.Tensor:
            labels = utils.tensor_to_np(labels)

        # if we're in sequence_mode
        if predictions.ndim == 3:
            # batch, classes, time
            if predictions.shape[1] < predictions.shape[2]:
                N, K, T = predictions.shape
                predictions = predictions.transpose(0, 2, 1).reshape(N * T, K)
            else:
                N, T, K = predictions.shape
                predictions = predictions.reshape(N * T, K)
        if labels.ndim == 3:
            if labels.shape[1] < labels.shape[2]:
                N, K, T = labels.shape
                labels = labels.transpose(0, 2, 1).reshape(N * T, K)
            else:
                N, T, K = labels.shape
                labels = labels.reshape(N * T, K)

        self.epoch_predictions.append(predictions)
        self.epoch_labels.append(labels)

    def compute_metrics_from_batches_labels(self, split):

        num_classes = self.epoch_predictions[0].shape[1]

        predictions = np.concatenate(self.epoch_predictions, axis=0).reshape(-1, num_classes)

        if self.epoch_labels[0].shape[-1] == num_classes:
            # if labels are one-hot
            labels = np.concatenate(self.epoch_labels, axis=0).reshape(-1, num_classes)
            # axis=1
            rows_with_false_labels = np.any(labels == self.ignore_index, axis=1)
            one_hot = True
        else:
            labels = np.concatenate(self.epoch_labels, axis=0).reshape(-1, )
            rows_with_false_labels = labels == self.ignore_index
            one_hot = False

        true_rows = np.logical_not(rows_with_false_labels)
        predictions = predictions[true_rows, :]
        labels = labels[true_rows, :] if one_hot else labels[true_rows]

        if self.evaluate_threshold:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics_by_threshold, epoch_metrics = evaluate_thresholds(predictions, labels, self.thresholds)
            with h5py.File(self.fname, 'r+') as f:
                for metric in self.threshold_metrics:
                    value = epoch_metrics[metric]
                    name = 'thresholds/' + split + '/' + metric
                    append_to_hdf5(f, name, value)
                # these are curves for each class for one single epoch
                for metric in self.threshold_curves:
                    name = 'threshold_curves/' + split + '/' + metric
                    if name in f:
                        del (f[name])
                    f.create_dataset(name, data=metrics_by_threshold[metric])

        # convert to intermediate representation for speed
        if one_hot:
            labels = onehot_to_index(labels)

        predictions = np.argmax(predictions, axis=1)

        with warnings.catch_warnings():
            data = {}
            for metric in self.metrics:
                if metric == 'confusion':
                    warnings.simplefilter("ignore")
                    data[metric] = confusion(predictions, labels, K=self.num_classes)
                    # import pdb
                    # pdb.set_trace()
                elif metric == 'binary_confusion':
                    pass
                else:
                    warnings.simplefilter("ignore")
                    data[metric] = self.metric_funcs[metric](predictions, labels)

        with h5py.File(self.fname, 'r+') as f:
            for metric, value in data.items():
                name = split + '/' + metric
                append_to_hdf5(f, name, value)
        if self.key_metric != 'loss':
            self.latest_key[split] = data[self.key_metric]


class OpticalFlow(Metrics):
    """ Metrics class for saving optic flow metrics to disk """
    def __init__(self, run_dir, key_metric, num_parameters, metrics=['SSIM_full'],
                 splits=['train', 'val']):
        super().__init__(run_dir, metrics, key_metric, 'opticalflow', num_parameters, splits)

    def compute_metrics_from_batches_labels(self, split):
        pass


def load_threshold_data(logger_file: Union[str, os.PathLike]) -> Tuple[dict, dict]:
    """ Convenience function for loading threshold data from a logger file. Useful for visualization

    Parameters
    ----------
    logger_file: str, os.PathLike
        path to a Classification metrics hdf5 file

    Returns
    -------
    metrics_by_threshold: dict
        see evaluate_thresholds
    epoch_summaries: dict
        see evaluate_thresholds
    """
    with h5py.File(logger_file, 'r') as f:
        metrics_by_threshold = {}
        epoch_summaries = {}
        dataset = f['threshold_curves']
        splits = list(dataset.keys())
        keys = dataset[splits[0]].keys()
        # print(splits)
        summary_keys = f['thresholds/train'].keys()

        for split in splits:
            metrics_by_threshold[split] = {}
            for key in keys:
                metrics_by_threshold[split][key] = f['threshold_curves/' + split + '/' + key][:]

        dataset = f['thresholds']
        splits = list(dataset.keys())
        # print(splits)
        summary_keys = dataset[splits[0]].keys()
        for split in splits:
            epoch_summaries[split] = {}
            for key in summary_keys:
                epoch_summaries[split][key] = f['thresholds/' + split + '/' + key][:]
        # print(metrics_by_threshold)
    return metrics_by_threshold, epoch_summaries
