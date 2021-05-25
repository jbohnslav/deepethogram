# from pytestqt import qtbot

from deepethogram.gui.main import run, setup_gui_cfg, MainWindow


def test_setup():
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