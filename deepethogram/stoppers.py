import logging
from typing import Type

from omegaconf import DictConfig

log = logging.getLogger(__name__)


class Stopper:
    """ Base class for stopping training """
    def __init__(self, name: str, start_epoch: int = 0, num_epochs: int = 1000):
        """ constructor for stopper

        Parameters
        ----------
        name: str
            name of the stopper. could be used by a training routine. E.g. if stopper.name == 'early': # do something
        start_epoch: int
            initializes epoch number. useful in case you want to pick up training from an exact state
        num_epochs: int
            number of epochs before training will automatically stop. used by subclasses
        """
        self.name = name
        self.epoch_counter = start_epoch
        self.num_epochs = num_epochs

    def step(self, *args, **kwargs):
        """ increment internal counter """
        self.epoch_counter += 1

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


class NumEpochsStopper(Stopper):
    def __init__(self, name: str = 'num_epochs', start_epoch: int = 0, num_epochs: int = 1000):
        super().__init__(name, start_epoch, num_epochs)

    def step(self, *args, **kwargs):
        super().step()
        should_stop = False
        if self.epoch_counter > self.num_epochs:
            should_stop = True
        return should_stop


class EarlyStopping(Stopper):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events
    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training
    modified from here
    https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py
    """

    def __init__(self, name='early', start_epoch=0, num_epochs=1000, patience=5, is_error=False,
                 early_stopping_begins: int = 0):
        super().__init__(name, start_epoch, num_epochs)
        if patience < 1:
            raise ValueError("Argument patience should be positive integer")
        self.patience = patience
        self.best_score = None
        self.is_error = is_error
        self.early_stopping_begins = early_stopping_begins

    def step(self, score):
        super().step()
        best = False
        should_stop = False
        # if the metric is actually an error, then we want to stop when validation error
        # stops SHRINKING rather than growing. Make it negative
        if self.is_error:
            score = -score
        if self.best_score is None:
            self.best_score = score
            best = True

        elif score < self.best_score:
            self.counter += 1
            # self._logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience and self.epoch_counter >= self.early_stopping_begins:
                print("EarlyStopping: Stop training")
                should_stop = True
        else:
            self.best_score = score
            self.counter = 0
            best = True
        if self.epoch_counter > self.num_epochs:
            should_stop = True
        return best, should_stop


class LearningRateStopper(Stopper):
    """Simple early stopper that stops when the learning rate drops below some threshold.
    Example usage: you reduce your learning rate when your validation loss stops improving. If the learning rate drops
        below some minimum value, automatically stop training.

    Example (pseudo-python):
        stopper = LearningRateStopper(5e-7)
        for i in range(num_epochs):
            train(model)
            if is_saturated(validation_loss):
                reduce_learning_rate(optimizer)
            if stopper(optimizer.learning_rate):
                break
    """

    def __init__(self, name='learning_rate', minimum_learning_rate: float = 5e-7, start_epoch=0, num_epochs=1000,
                 eps: float = 1e-8):
        super().__init__(name, start_epoch, num_epochs)
        """Constructor for LearningRateStopper.
        Args:
            minimum_learning_rate: if learning rate drops below this value, automatically stop training
        """
        self.minimum_learning_rate = minimum_learning_rate
        self.eps = eps

    def step(self, lr: float) -> bool:
        """Computes if learning rate is below the set value
        Args:
            lr (float): learning rate
        Returns:
            should_stop: whether or not to stop training
        """
        super().step()
        should_stop = False
        # print('epoch counter: {} num_epochs: {}'.format(self.epoch_counter, self.num_epochs))
        if lr < self.minimum_learning_rate + self.eps or self.epoch_counter >= self.num_epochs:
            print('Reached learning rate {}, stopping...'.format(lr))
            should_stop = True
        return should_stop


def get_stopper(cfg: DictConfig) -> Type[Stopper]:
    """

    Parameters
    ----------
    cfg

    Returns
    -------
    stopper: subclass of stoppers.Stopper

    """
    # ASSUME WE'RE USING LOSS AS THE KEY METRIC, WHICH IS AN ERROR
    log.info('Using stopper type {}'.format(cfg.train.stopping_type))
    if cfg.train.stopping_type == 'early':
        return EarlyStopping(start_epoch=0, num_epochs=cfg.train.num_epochs,
                             patience=cfg.train.patience,
                             is_error=True, early_stopping_begins=cfg.train.early_stopping_begins)
    elif cfg.train.stopping_type == 'learning_rate':
        return LearningRateStopper(start_epoch=0, num_epochs=cfg.train.num_epochs,
                                   minimum_learning_rate=cfg.train.min_lr)
    else:
        return NumEpochsStopper('num_epochs', start_epoch=0, num_epochs=cfg.train.num_epochs)
