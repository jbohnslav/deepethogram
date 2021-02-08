from collections import defaultdict
import logging
import os
import time

from pytorch_lightning.callbacks import Callback

from deepethogram import utils

log = logging.getLogger(__name__)

class DebugCallback(Callback):
    def __init__(self):
        super().__init__()
        log.info('callback initialized')

    def on_init_end(self, trainer):
        log.info('on init start')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        log.debug('on train batch start')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        log.debug('on train batch end')

    def on_train_epoch_start(self, trainer, pl_module):
        log.info('on train epoch start')

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        log.info('on train epoch end')

    def on_validation_epoch_start(self, trainer, pl_module):
        log.info('on validation epoch start')

    def on_validation_epoch_end(self, trainer, pl_module):
        log.info('on validation epoch end')

    def on_test_epoch_start(self, trainer, pl_module):
        log.info('on test epoch start')

    def on_test_epoch_end(self, trainer, pl_module):
        log.info('on test epoch end')

    def on_epoch_start(self, trainer, pl_module):
        log.info('on epoch start')

    def on_epoch_end(self, trainer, pl_module):
        log.info('on epoch end')

    def on_train_start(self, trainer, pl_module):
        log.info('on train start')

    def on_train_end(self, trainer, pl_module):
        log.info('on train end')

    def on_validation_start(self, trainer, pl_module):
        log.info('on validation start')

    def on_validation_end(self, trainer, pl_module):
        log.info('on validation end')

    def on_keyboard_interrupt(self, trainer, pl_module):
        log.info('on keyboard interrupt')

# class FPSCallback(Callback):
#     def on_init_start(self, trainer):
#         self.times = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'speedtest': 0.0}
#         self.n_images = {'train': 0, 'val': 0, 'test': 0, 'speedtest': 0}
#         self.fps = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'speedtest': 0.0}
#         # print('Starting to init trainer!')
#         pass
#
#     def get_num_images(self, batch):
#         keys = list(batch.keys())
#         batch_size = batch[keys[0]].shape[0]
#         return batch_size
#
#     def on_train_epoch_start(self, trainer, pl_module):
#         self.times['train'] = time.time()
#
#     def on_validation_epoch_start(self, trainer, pl_module):
#         self.times['val'] = time.time()
#
#     def on_test_epoch_start(self, trainer, pl_module):
#         self.times['test'] = time.time()
#
#     def get_fps(self, split):
#         elapsed = time.time() - self.times[split]
#         n_images = self.n_images[split]
#         return n_images / elapsed
#
#     def on_train_epoch_end(self, trainer, pl_module, outputs):
#         pl_module.metrics.buffer.append('train', {'fps': self.get_fps('train')})
#         # import pdb; pdb.set_trace()
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         pl_module.metrics.buffer.append('val', {'fps': self.get_fps('val')})
#
#     def on_test_epoch_end(self, trainer, pl_module):
#         pl_module.metrics.buffer.append('test', {'fps': self.get_fps('test')})
#
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         self.n_images['train'] += self.get_num_images(batch)
#
#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         self.n_images['val'] += self.get_num_images(batch)
#
#     def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         self.n_images['test'] += self.get_num_images(batch)


class FPSCallback(Callback):
    def __init__(self):
        super().__init__()
        self.times = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'speedtest': 0.0}
        self.n_images = {'train': 0, 'val': 0, 'test': 0, 'speedtest': 0}
        self.fps = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'speedtest': 0.0}

    def start_timer(self, split):
        self.times[split] = time.time()
        self.n_images[split] = 0

    def get_num_images(self, batch):
        keys = list(batch.keys())
        batch_size = batch[keys[0]].shape[0]
        return batch_size

    def end_batch(self, split, batch, pl_module):
        elapsed = time.time() - self.times[split]
        n_images = self.get_num_images(batch)
        fps = n_images / elapsed

        pl_module.metrics.buffer.append(split, {'fps': fps})

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.start_timer('train')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.end_batch('train', batch, pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.start_timer('val')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.end_batch('val', batch, pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.start_timer('speedtest')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.end_batch('speedtest', batch, pl_module)


# class SpeedtestCallback(Callback):
#     def __init__(self):
#         super().__init__()
#
#     def on_validation_end(self, trainer, pl_module):
#         trainer.test(pl_module)


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.metrics.buffer.append('train', {'lr': utils.get_minimum_learning_rate(pl_module.optimizer)})
        pl_module.metrics.end_epoch('train')
        latest_key = pl_module.metrics.latest_key['train']
        key = 'train_{}'.format(pl_module.metrics.key_metric)
        pl_module.log(key, latest_key, on_epoch=True)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.metrics.end_epoch('val')
        latest_key = pl_module.metrics.latest_key['val']
        key = 'val_{}'.format(pl_module.metrics.key_metric)
        pl_module.log(key, latest_key, on_epoch=True)

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.metrics.end_epoch('test')
        # pl_module.metrics.end_epoch('speedtest')

    def on_keyboard_interrupt(self, trainer, pl_module):
        pl_module.metrics.buffer.clear()


class ExampleImagesCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        # unfortunate hack: add viz_cnt dict to the lightning module itself
        # this way, in the visualize_batch method of the lightning module, we know whether to visualize or to
        # skip it
        pl_module.viz_cnt = defaultdict(int)

    def reset_cnt(self, pl_module, split):
        pl_module.viz_cnt[split] = 0

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        self.reset_cnt(pl_module, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.reset_cnt(pl_module, 'val')

    def on_test_epoch_end(self, trainer, pl_module):
        self.reset_cnt(pl_module, 'test')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pl_module.viz_cnt['train'] += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pl_module.viz_cnt['val'] += 1

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pl_module.viz_cnt['test'] += 1

class CheckpointCallback(Callback):
    def __init__(self):
        super().__init__()

    def checkpoint(self, pl_module):
        utils.checkpoint(pl_module.model, os.getcwd(), pl_module.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        self.checkpoint(pl_module)

    def on_keyboard_interrupt(self, trainer, pl_module):
        self.checkpoint(pl_module)


class StopperCallback(Callback):
    def __init__(self, stopper):
        super().__init__()
        self.stopper = stopper
        # self.should_stop = False

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # do this when starting because we're sure that both validation and training have ended
        if pl_module.current_epoch == 0:
            return

        if self.stopper.name == 'early':
            _, should_stop = self.stopper(pl_module.metrics.latest_key['val'])
        elif self.stopper.name == 'learning_rate':
            min_lr = pl_module.metrics[('train', 'lr', -1)]
            # log.info('LR: {}'.format(min_lr))
            should_stop = self.stopper(min_lr)
        elif self.stopper.name == 'num_epochs':
            should_stop = self.stopper.step()
        else:
            raise ValueError('invalid stopping name: {}'.format(self.stopper.name))

        if should_stop:
            # log.info('Stopping criterion reached! Raising KeyboardInterrupt to quit')
            log.info('Stopping criterion reached! setting trainer.should_stop=True')
            trainer.should_stop = True
            # raise KeyboardInterrupt
