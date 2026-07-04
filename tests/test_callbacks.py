import os
from types import SimpleNamespace

from deepethogram import callbacks
from deepethogram.callbacks import CheckpointCallback, MetricsCallback


class Buffer:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


def test_metrics_callback_clears_buffer_on_keyboard_interrupt():
    buffer = Buffer()
    pl_module = SimpleNamespace(metrics=SimpleNamespace(buffer=buffer))

    MetricsCallback().on_exception(None, pl_module, KeyboardInterrupt())

    assert buffer.cleared


def test_metrics_callback_ignores_other_exceptions():
    buffer = Buffer()
    pl_module = SimpleNamespace(metrics=SimpleNamespace(buffer=buffer))

    MetricsCallback().on_exception(None, pl_module, RuntimeError("boom"))

    assert not buffer.cleared


def test_checkpoint_callback_saves_on_keyboard_interrupt(monkeypatch):
    calls = []
    model = object()
    pl_module = SimpleNamespace(model=model, current_epoch=7)

    def checkpoint(model, directory, epoch):
        calls.append((model, directory, epoch))

    monkeypatch.setattr(callbacks.utils, "checkpoint", checkpoint)

    CheckpointCallback().on_exception(None, pl_module, KeyboardInterrupt())

    assert len(calls) == 1
    assert calls[0][0] is model
    assert calls[0][1] == os.getcwd()
    assert calls[0][2] == 7


def test_checkpoint_callback_ignores_other_exceptions(monkeypatch):
    calls = []
    pl_module = SimpleNamespace(model=object(), current_epoch=7)

    monkeypatch.setattr(callbacks.utils, "checkpoint", lambda *args: calls.append(args))

    CheckpointCallback().on_exception(None, pl_module, RuntimeError("boom"))

    assert calls == []
