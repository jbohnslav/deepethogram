# from pytestqt import qtbot
import os

import pytest


@pytest.mark.skipif(os.environ['DEG_VERSION'] == 'headless', reason="Dont run GUI tests for headless deepethogram")
def test_setup():
    # put imports here so that headless version does not import gui tools
    from deepethogram.gui.main import run, setup_gui_cfg, MainWindow
    cfg = setup_gui_cfg()

    assert cfg.run.type == 'gui'


# def test_new_project():
#     cfg = setup_gui_cfg()

#     window = MainWindow(cfg)
#     window.resize(1024, 768)
#     window.show()

#     window._new_project()
# def test_open():
#     run()