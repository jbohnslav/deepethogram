# from pytestqt import qtbot
import os

import pytest

DEG_VERSION = os.environ.get("DEG_VERSION", "full")


@pytest.mark.skipif(
    DEG_VERSION == "headless",
    reason="Dont run GUI tests for headless deepethogram",
)
def test_setup():
    # put imports here so that headless version does not import gui tools
    from deepethogram.gui.main import MainWindow, run, setup_gui_cfg

    cfg = setup_gui_cfg()

    assert cfg.run.type == "gui"


# def test_new_project():
#     cfg = setup_gui_cfg()

#     window = MainWindow(cfg)
#     window.resize(1024, 768)
#     window.show()

#     window._new_project()
# def test_open():
#     run()
