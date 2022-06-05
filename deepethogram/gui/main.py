import logging
import os
import subprocess
import sys
import traceback
from functools import partial
from typing import Union

import cv2
# import hydra
import numpy as np
import pandas as pd
from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtCore import Slot
from PySide2.QtWidgets import (QMainWindow, QFileDialog, QInputDialog)
from omegaconf import DictConfig, OmegaConf

from deepethogram import projects, utils, configuration
from deepethogram.file_io import VideoReader
from deepethogram.postprocessing import get_postprocessor_from_cfg
from deepethogram.gui.custom_widgets import UnclickButtonOnPipeCompletion, SubprocessChainer
from deepethogram.gui.mainwindow import Ui_MainWindow
from deepethogram.gui.menus_and_popups import CreateProject, simple_popup_question, ShouldRunInference, overwrite_or_not

log = logging.getLogger(__name__)

pretrained_models_error = 'Dont train flow generator without pretrained weights! ' + \
            'See the project GitHub for instructions on downloading weights: https://github.com/jbohnslav/deepethogram'


class MainWindow(QMainWindow):
    # lots of inspiration from this wonderful stackoverflow answer here:
    # https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview/35514531#35514531
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg  # has UI elements, like label view width, how far to jump with ctrl-arrow, colormaps, etc

        self.debug = log.isEnabledFor(logging.DEBUG)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('DeepEthogram')

        # print(dir(self.ui.actionOpen))
        self.ui.videoBox.setLayout(self.ui.formLayout)
        self.ui.actionOpen.triggered.connect(self.open_avi_browser)
        # self.ui.plainTextEdit.textChanged.connect(self.text_change)
        self.ui.actionAdd.triggered.connect(self.add_class)
        self.ui.actionRemove.triggered.connect(self.remove_class)
        self.ui.actionNew_Project.triggered.connect(self._new_project)
        self.ui.actionOpen_Project.triggered.connect(self.load_project)
        self.ui.flow_train.clicked.connect(self.flow_train)
        self.ui.featureextractor_train.clicked.connect(self.featureextractor_train)
        self.ui.featureextractor_infer.clicked.connect(self.featureextractor_infer)
        self.ui.sequence_train.clicked.connect(self.sequence_train)
        self.ui.sequence_infer.clicked.connect(self.sequence_infer)
        self.ui.importLabels.triggered.connect(self.import_external_labels)
        self.ui.importPredictions.clicked.connect(self.import_predictions_as_labels)
        self.ui.classifierInference.triggered.connect(self.classifier_inference)
        self.ui.actionAdd_multiple.triggered.connect(self.add_multiple_videos)
        self.ui.actionOvernight.triggered.connect(self.run_overnight)
        self.ui.predictionsCombo.currentTextChanged.connect(self.change_predictions)

        self.default_archs = None

        # scroll down to Standard Shorcuts to find what the keys are called:
        # https://doc.qt.io/qt-5/qkeysequence.html
        next_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Right'), self)
        # partial functions create a new, separate function with certain default arguments
        next_shortcut.activated.connect(partial(self.move_n_frames, 1))
        next_shortcut.activated.connect(self.user_did_something)

        up_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Up'), self)
        up_shortcut.activated.connect(partial(self.move_n_frames, -cfg.vertical_arrow_jump))
        up_shortcut.activated.connect(self.user_did_something)

        down_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Down'), self)
        down_shortcut.activated.connect(partial(self.move_n_frames, cfg.vertical_arrow_jump))
        down_shortcut.activated.connect(self.user_did_something)

        back_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Left'), self)
        back_shortcut.activated.connect(partial(self.move_n_frames, -1))
        back_shortcut.activated.connect(self.user_did_something)

        jumpleft_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Left'), self)
        jumpleft_shortcut.activated.connect(partial(self.move_n_frames, -cfg.control_arrow_jump))
        jumpleft_shortcut.activated.connect(self.user_did_something)

        jumpright_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Right'), self)
        jumpright_shortcut.activated.connect(partial(self.move_n_frames, cfg.control_arrow_jump))
        jumpright_shortcut.activated.connect(self.user_did_something)

        self.ui.actionSave_Project.triggered.connect(self.save)
        save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self)
        save_shortcut.activated.connect(self.save)
        finalize_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+F'), self)
        finalize_shortcut.activated.connect(self.finalize)
        open_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
        open_shortcut.activated.connect(self.load_project)
        self.ui.finalize_labels.clicked.connect(self.finalize)
        self.ui.exportPredictions.clicked.connect(self.export_predictions)
        self.saved = True
        self.num_label_initializations = 0

        self.toggle_shortcuts = []

        self.initialized_label = False
        self.initialized_prediction = False
        self.data_path = None
        self.model_path = None
        self.latent_name = None
        self.thresholds = None
        for i in range(10):
            self.toggle_shortcuts.append(QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), self))
            tmp_func = partial(self.respond_to_keypress, i)
            self.toggle_shortcuts[i].activated.connect(tmp_func)
            self.toggle_shortcuts[i].activated.connect(self.user_did_something)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(60 * 1000)
        self.timer.timeout.connect(self.log_idle)
        self.timer.start()

        # the current directory is where_user_launched/gui_logs/date_time_runlog
        initialized_directory = os.path.dirname(os.path.dirname(os.getcwd()))
        if os.path.isfile(os.path.join(initialized_directory, 'project_config.yaml')):
            self.initialize_project(initialized_directory)

        # log.info('children: {}'.format(self.children()))
        self.show()

    def user_did_something(self):
        if self.timer.isActive():
            pass
        else:
            # else, the user was already idle
            # will have a timestamp by the logfile
            log.info('User restarted labeling')
        self.timer.start()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # print('key pressed')
        self.user_did_something()
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.user_did_something()
        super().mousePressEvent(event)

    def log_idle(self):
        log.info('User has been idle for {} seconds...'.format(float(self.timer.interval()) / 1000))
        self.timer.stop()

    def respond_to_keypress(self, keynum: int):
        # print('key pressed')
        if self.ui.labels.label is not None:
            self.ui.labels.label.toggle_behavior(keynum)
        else:
            # print('none')
            return

    def has_trained(self, model_type: str) -> bool:
        weight_dict = projects.get_weights_from_model_path(self.cfg.project.model_path)
        model_list = weight_dict[model_type]
        if len(model_list) == 0:
            return False
        else:
            return True

    def project_loaded_buttons(self):
        self.ui.actionOpen.setEnabled(True)
        self.ui.actionAdd.setEnabled(True)
        self.ui.actionRemove.setEnabled(True)
        number_finalized_labels = projects.get_number_finalized_labels(self.cfg)
        log.info('Number finalized labels: {}'.format(number_finalized_labels))
        if self.has_trained('flow_generator'):
            # self.ui.flow_inference.setEnabled(True)
            self.ui.flow_train.setEnabled(True)
        if self.has_trained('feature_extractor') or number_finalized_labels > 1:
            self.ui.featureextractor_infer.setEnabled(True)
            self.ui.featureextractor_train.setEnabled(True)

        # require for sequence training that there be non-zero number of output files
        n_output_files = 0
        if self.data_path is not None:
            records = projects.get_records_from_datadir(self.data_path)
            for animal, record in records.items():
                if record['output'] is not None and os.path.isfile(record['output']):
                    n_output_files += 1

        if self.has_trained('feature_extractor') and n_output_files > 2:
            self.ui.sequence_train.setEnabled(True)

        if self.has_trained('sequence') and n_output_files > 0:
            self.ui.sequence_infer.setEnabled(True)
            self.ui.classifierInference.setEnabled(True)

        self.unfinalized = projects.get_unfinalized_records(self.cfg)
        self.unfinalized_idx = 0

    def video_loaded_buttons(self):
        self.ui.finalize_labels.setEnabled(True)
        self.ui.actionSave_Project.setEnabled(True)
        records = projects.get_records_from_datadir(self.data_path)
        if len(records) > 1:
            self.ui.flow_train.setEnabled(True)

    def initialize_video(self, videofile: Union[str, os.PathLike]):
        if hasattr(self, 'vid'):
            self.vid.close()
            # if hasattr(self.vid, 'cap'):
            #     self.vid.cap.release()
        self.videofile = videofile
        try:
            self.ui.videoPlayer.videoView.initialize_video(videofile)
            # for convenience extract the videoplayer object out of the videoView
            self.vid = self.ui.videoPlayer.videoView.vid
            # for convenience
            self.n_timepoints = len(self.ui.videoPlayer.videoView.vid)

            log.debug('is deg: {}'.format(projects.is_deg_file(videofile)))

            if os.path.normpath(
                    self.cfg.project.data_path) in os.path.normpath(videofile) and projects.is_deg_file(videofile):
                record = projects.get_record_from_subdir(os.path.dirname(videofile))
                log.info('Record for loaded video: {}'.format(record))
                labelfile = record['label']
                outputfile = record['output']
                self.labelfile = labelfile
                self.outputfile = outputfile
                if labelfile is not None:
                    self.import_labelfile(labelfile)
                else:
                    self.initialize_label()
                if outputfile is not None:
                    self.import_outputfile(outputfile, first_time=True)
                else:
                    self.ui.predictionsCombo.clear()
                    self.initialize_prediction()
            else:
                log.info('Copying {} to your DEG directory'.format(videofile))
                new_loc = projects.add_video_to_project(OmegaConf.to_container(self.cfg), videofile)
                log.debug('New video location: {}'.format(new_loc))
                self.videofile = new_loc
                log.debug('New record: {}'.format(
                    utils.load_yaml(os.path.join(os.path.dirname(self.videofile), 'record.yaml'))))
                self.initialize_label()
                self.initialize_prediction()
            self.video_loaded_buttons()
        except BaseException as e:
            log.exception('Error initializing video: {}'.format(e))
            tb = traceback.format_exc()
            print(tb)
            return
        self.ui.videoPlayer.videoView.update_frame(0, force=True)
        self.setWindowTitle('DeepEthogram: {}'.format(self.cfg.project.name))
        self.update_video_info()
        self.user_did_something()

    def update_video_info(self):
        name = os.path.basename(self.videofile)
        nframes = self.n_timepoints

        with VideoReader(self.videofile) as reader:
            try:
                fps = reader.fps
                duration = nframes / fps
                fps = '{:.2f}'.format(fps)
                duration = '{:.2f}'.format(duration)
            except:
                fps = 'N/A'
                duration = 'N/A'
        num_labeled = self.ui.labels.label.changed.sum()

        self.ui.nameLabel.setText(name)
        self.ui.nframesLabel.setText('{:,}'.format(nframes))
        self.ui.nlabeledLabel.setText('{:,}'.format(num_labeled))
        self.ui.nunlabeledLabel.setText('{:,}'.format(nframes - num_labeled))
        self.ui.durationLabel.setText(duration)
        self.ui.fpsLabel.setText(fps)

        self.ui.labels.label.num_changed.connect(self.update_num_labeled)

    def update_num_labeled(self, n: Union[int, str]):
        self.ui.nlabeledLabel.setText('{:,}'.format(n))
        self.ui.nunlabeledLabel.setText('{:,}'.format(self.n_timepoints - n))

    def initialize_label(self, label_array: np.ndarray = None, debug: bool = False):
        if self.cfg.project is None:
            raise ValueError('must load or create project before initializing a label!')
        self.ui.labels.initialize(behaviors=OmegaConf.to_container(self.cfg.project.class_names),
                                  n_timepoints=self.n_timepoints,
                                  debug=debug,
                                  fixed=False,
                                  array=label_array,
                                  colormap=self.cfg.cmap)
        # we never want to connect signals to slots more than once
        log.debug('initialized label: {}'.format(self.initialized_label))
        # if not self.initialized_label:
        self.ui.videoPlayer.videoView.frameNum.connect(self.ui.labels.label.change_view_x)
        self.ui.labels.label.saved.connect(self.update_saved)
        self.initialized_label = True
        self.update()

    def initialize_prediction(self,
                              prediction_array: np.ndarray = None,
                              debug: bool = False,
                              opacity: np.ndarray = None):

        # do all the setup for labels and predictions
        self.ui.predictions.initialize(behaviors=OmegaConf.to_container(self.cfg.project.class_names),
                                       n_timepoints=self.n_timepoints,
                                       debug=debug,
                                       fixed=True,
                                       array=prediction_array,
                                       opacity=opacity,
                                       colormap=self.cfg.cmap)
        # if not self.initialized_prediction:
        self.ui.videoPlayer.videoView.frameNum.connect(self.ui.predictions.label.change_view_x)
        # we don't want to be able to manually edit the predictions
        self.ui.predictions.buttons.fix()
        self.initialized_prediction = True
        self.update()

    def generate_flow_train_args(self):
        args = ['python', '-m', 'deepethogram.flow_generator.train', 'project.path={}'.format(self.cfg.project.path)]
        weights = self.get_selected_models()['flow_generator']
        if weights is None:
            raise ValueError(pretrained_models_error)
        if weights is not None and os.path.isfile(weights):
            args += ['reload.weights={}'.format(weights)]
        return args

    def flow_train(self):
        if self.ui.flow_train.isChecked():
            # self.ui.flow_inference.setEnabled(False)
            self.ui.featureextractor_train.setEnabled(False)
            self.ui.featureextractor_infer.setEnabled(False)
            self.ui.sequence_infer.setEnabled(False)
            self.ui.sequence_train.setEnabled(False)

            args = self.generate_flow_train_args()
            log.info('flow_train called with args: {}'.format(args))
            self.training_pipe = subprocess.Popen(args)
            self.listener = UnclickButtonOnPipeCompletion(self.ui.flow_train, self.training_pipe)
            self.listener.start()
        else:
            if self.training_pipe.poll() is None:
                self.training_pipe.terminate()
                self.training_pipe.wait()
                log.info('Training interrupted.')
            else:
                log.info('Training finished. If you see error messages above, training did not complete successfully.')
            # self.train_thread.terminate()
            del self.training_pipe
            self.listener.quit()
            self.listener.wait()
            del self.listener
            log.info('~' * 100)

            self.project_loaded_buttons()
            self.get_trained_models()

    def featureextractor_train(self):
        if self.ui.featureextractor_train.isChecked():
            self.ui.flow_train.setEnabled(False)
            # self.ui.flow_inference.setEnabled(False)
            self.ui.featureextractor_infer.setEnabled(False)
            self.ui.sequence_infer.setEnabled(False)
            self.ui.sequence_train.setEnabled(False)

            args = [
                'python', '-m', 'deepethogram.feature_extractor.train', 'project.path={}'.format(self.cfg.project.path)
            ]
            print(self.get_selected_models())
            weights = self.get_selected_models()['feature_extractor']
            # print(weights)
            if weights is None:
                raise ValueError(pretrained_models_error)
            if os.path.isfile(weights):
                args += ['feature_extractor.weights={}'.format(weights)]
            flow_weights = self.get_selected_models()['flow_generator']  # ('flow_generator')
            assert flow_weights is not None
            args += ['flow_generator.weights={}'.format(flow_weights)]
            log.info('feature extractor train called with args: {}'.format(args))
            self.training_pipe = subprocess.Popen(args)
            self.listener = UnclickButtonOnPipeCompletion(self.ui.featureextractor_train, self.training_pipe)
            self.listener.start()
        else:
            if self.training_pipe.poll() is None:
                self.training_pipe.terminate()
                self.training_pipe.wait()
                log.info('Training interrupted.')
            else:
                log.info('Training finished. If you see error messages above, training did not complete successfully.')
            del self.training_pipe
            self.listener.quit()
            self.listener.wait()
            del self.listener
            log.info('~' * 100)
            # self.ui.flow_train.setEnabled(True)
            self.project_loaded_buttons()
            self.get_trained_models()
            # self.listener.stop()

            # self.ui.featureextractor_infer.setEnabled(True)

    def generate_featureextractor_inference_args(self):
        records = projects.get_records_from_datadir(self.data_path)
        keys, no_outputs = [], []
        for key, record in records.items():
            keys.append(key)
            no_outputs.append(record['output'] is None)
        form = ShouldRunInference(keys, no_outputs)
        ret = form.exec_()
        if not ret:
            return
        should_infer = form.get_outputs()
        all_false = np.all(np.array(should_infer) == False)
        if all_false:
            return
        self.ui.flow_train.setEnabled(False)
        # self.ui.flow_inference.setEnabled(False)
        self.ui.featureextractor_train.setEnabled(False)
        weights = self.get_selected_models()['feature_extractor']
        if weights is not None and os.path.isfile(weights):
            weight_arg = 'feature_extractor.weights={}'.format(weights)
        else:
            raise ValueError('Dont run inference without using a proper feature extractor weights! {}'.format(weights))

        args = [
            'python', '-m', 'deepethogram.feature_extractor.inference', 'project.path={}'.format(self.cfg.project.path),
            'inference.overwrite=True', weight_arg
        ]
        flow_weights = self.get_selected_models()['flow_generator']
        assert flow_weights is not None
        args += ['flow_generator.weights={}'.format(flow_weights)]
        string = 'inference.directory_list=['
        for key, infer in zip(keys, should_infer):
            if infer:
                record_dir = os.path.join(self.data_path, key) + ','
                string += record_dir
        string = string[:-1] + ']'
        args += [string]
        return args

    def featureextractor_infer(self):
        if self.ui.featureextractor_infer.isChecked():
            args = self.generate_featureextractor_inference_args()
            if args is None:
                return
            log.info('inference running with args: {}'.format(' '.join(args)))
            self.inference_pipe = subprocess.Popen(args)
            self.listener = UnclickButtonOnPipeCompletion(self.ui.featureextractor_infer, self.inference_pipe)
            self.listener.start()
        else:
            if self.inference_pipe.poll() is None:
                self.inference_pipe.terminate()
                self.inference_pipe.wait()
                log.info('Inference interrupted.')
            else:
                log.info(
                    'Inference finished. If you see error messages above, inference did not complete successfully.')
            del self.inference_pipe
            self.listener.quit()
            self.listener.wait()
            del self.listener
            log.info('~' * 100)
            self.project_loaded_buttons()
            record = projects.get_record_from_subdir(os.path.dirname(self.videofile))
            if record['output'] is not None:
                self.outputfile = record['output']
            else:
                self.outputfile = None
            self.import_outputfile(self.outputfile)

            # self.ui.featureextractor_infer.setEnabled(True)

    def sequence_train(self):
        if self.ui.sequence_train.isChecked():
            self.ui.flow_train.setEnabled(False)
            # self.ui.flow_inference.setEnabled(False)
            self.ui.featureextractor_train.setEnabled(False)
            self.ui.featureextractor_infer.setEnabled(False)
            self.ui.sequence_infer.setEnabled(False)
            # self.ui.sequence_train.setEnabled(False)
            args = ['python', '-m', 'deepethogram.sequence.train', 'project.path={}'.format(self.cfg.project.path)]
            weights = self.get_selected_models()['sequence']
            if weights is not None and os.path.isfile(weights):
                args += ['reload.weights={}'.format(weights)]
            self.training_pipe = subprocess.Popen(args)
            self.listener = UnclickButtonOnPipeCompletion(self.ui.sequence_train, self.training_pipe)
            self.listener.start()
        else:
            # self.train_thread.terminate()
            if self.training_pipe.poll() is None:
                self.training_pipe.terminate()
                self.training_pipe.wait()
                log.info('Training interrupted.')
            else:
                log.info('Training finished. If you see error messages above, training did not complete successfully.')
            del self.training_pipe
            self.listener.quit()
            self.listener.wait()
            del self.listener
            log.info('~' * 100)
            # self.ui.flow_train.setEnabled(True)
            self.project_loaded_buttons()
            self.get_trained_models()
            # self.ui.featureextractor_infer.setEnabled(True)

    def generate_sequence_inference_args(self):
        records = projects.get_records_from_datadir(self.data_path)
        keys = list(records.keys())
        outputs = projects.has_outputfile(records)
        sequence_weights = self.get_selected_models()['sequence']
        if sequence_weights is not None and os.path.isfile(sequence_weights):
            run_files = utils.get_run_files_from_weights(sequence_weights)
            sequence_config = OmegaConf.load(run_files['config_file'])
            # sequence_config = utils.load_yaml(os.path.join(os.path.dirname(sequence_weights), 'config.yaml'))
            latent_name = sequence_config['sequence']['latent_name']
            if latent_name is None:
                latent_name = sequence_config['feature_extractor']['arch']
            output_name = sequence_config['sequence']['output_name']
            if output_name is None:
                output_name = sequence_config['sequence']['arch']
        else:
            raise ValueError('must specify a valid weight file to run sequence inference!')

        log.debug('latent name: {}'.format(latent_name))
        # sequence_name, _ = utils.get_latest_model_and_name(self.project_config['project']['path'], 'sequence')

        # GOAL: MAKE ONLY FILES WITH LATENT_NAME PRESENT APPEAR ON LIST
        # SHOULD BE UNCHECKED IF THERE IS ALREADY THE "OUTPUT NAME" IN FILE

        has_latents = projects.do_outputfiles_have_predictions(self.data_path, latent_name)
        has_outputs = projects.do_outputfiles_have_predictions(self.data_path, output_name)
        no_sequence_outputs = [outputs[i] and not has_outputs[i] for i in range(len(records))]
        keys_with_features = []
        for i, key in enumerate(keys):
            if has_latents[i]:
                keys_with_features.append(key)
        form = ShouldRunInference(keys_with_features, no_sequence_outputs)
        ret = form.exec_()
        if not ret:
            return
        should_infer = form.get_outputs()
        all_false = np.all(np.array(should_infer) == False)
        if all_false:
            return
        weights = self.get_selected_models()['sequence']
        if weights is not None and os.path.isfile(weights):
            weight_arg = 'sequence.weights={}'.format(weights)
        else:
            raise ValueError('weights do not exist! {}'.format(weights))
        args = [
            'python', '-m', 'deepethogram.sequence.inference', 'project.path={}'.format(self.cfg.project.path),
            'inference.overwrite=True', weight_arg
        ]
        string = 'inference.directory_list=['
        for key, infer in zip(keys, should_infer):
            if infer:
                record_dir = os.path.join(self.data_path, key) + ','
                string += record_dir
        string = string[:-1] + ']'
        args += [string]
        return args

    def sequence_infer(self):
        if self.ui.sequence_infer.isChecked():

            args = self.generate_sequence_inference_args()
            if args is None:
                return
            log.info('sequence inference running with args: {}'.format(args))
            self.inference_pipe = subprocess.Popen(args)
            self.listener = UnclickButtonOnPipeCompletion(self.ui.sequence_infer, self.inference_pipe)
            self.listener.start()
        else:
            if self.inference_pipe.poll() is None:
                self.inference_pipe.terminate()
                self.inference_pipe.wait()
                log.info('Inference interrupted.')
            else:
                log.info('Inference finished')
            log.info('~' * 100)
            del self.inference_pipe
            self.listener.quit()
            self.listener.wait()
            del self.listener
            self.project_loaded_buttons()
            # del self.listener
            record = projects.get_record_from_subdir(os.path.dirname(self.videofile))
            if record['output'] is not None:
                self.outputfile = record['output']
            else:
                self.outputfile = None
            self.import_outputfile(self.outputfile, first_time=True)

    def classifier_inference(self):
        if self.ui.classifierInference.isChecked():
            fe_args = self.generate_featureextractor_inference_args()
            sequence_args = self.generate_sequence_inference_args()

            if fe_args is None or sequence_args is None:
                log.error('Erroneous arguments to fe or seq: {}, {}'.format(fe_args, sequence_args))

            calls = [fe_args, sequence_args]

            # calls = [['ping', 'localhost', '-n', '10'], ['dir']]
            self.listener = SubprocessChainer(calls)
            self.listener.start()
            # self.inference_thread = utils.chained_subprocess_calls(calls)
        else:
            self.listener.stop()
            self.listener.wait()
            del self.listener
            self.project_loaded_buttons()
            # self.inference_thread
            # print(should_be_checked)

    def run_overnight(self):
        if self.ui.actionOvernight.isChecked():
            flow_args = self.generate_flow_train_args()
            fe_args = self.generate_featureextractor_inference_args()
            sequence_args = self.generate_sequence_inference_args()

            if flow_args is None:
                log.error('Erroneous flow arguments in run overnight: {}'.format(flow_args))
            if fe_args is None:
                log.error('Erroneous fe arguments in run overnight: {}'.format(fe_args))
            if sequence_args is None:
                log.error('Erroneous seq arguments in run overnight: {}'.format(sequence_args))
            calls = [flow_args, fe_args, sequence_args]

            # calls = [['ping', 'localhost', '-n', '10'], ['dir']]
            self.listener = SubprocessChainer(calls)
            self.listener.start()
            # self.inference_thread = utils.chained_subprocess_calls(calls)
        else:
            self.listener.stop()
            self.listener.wait()
            del self.listener
            self.project_loaded_buttons()
            # self.inference_thread
            # print(should_be_checked)

    def _new_project(self):

        form = CreateProject()
        ret = form.exec_()
        if not ret:
            return
        project_name = form.project_box.text()
        if project_name == form.project_name_default:
            log.warning('Must change project name')
            return

        labeler = form.labeler_box.text()
        if labeler == form.label_default_string:
            log.warning('Must specify a labeler')
            return
        behaviors = form.behaviors_box.text()
        if behaviors == form.behavior_default_string:
            log.warning('Must add list of behaviors')
            return
        project_name = project_name.replace(' ', '_')
        labeler = labeler.replace(' ', '_')
        behaviors = behaviors.replace(' ', '')
        behaviors = behaviors.split(',')
        behaviors.insert(0, 'background')

        project_dict = projects.initialize_project(form.project_directory, project_name, behaviors, labeler)

        self.initialize_project(project_dict['project']['path'])

    def add_class(self):
        text, ok = QInputDialog.getText(self, 'AddBehaviorDialog', 'Enter behavior name: ')
        if len(text) == 0:
            log.warning('No behavior entered')
            ok = False
        if not ok:
            return

        if text in self.cfg.project.class_names:
            log.warning('This behavior is already in the list...')
            return
            # self.add_class()
        text = text.replace(' ', '_')
        log.info('new behavior name: {}'.format(text))

        message = '''Are you sure you want to add behavior {}? 
            All previous labels will have a blank column added that must be labeled.
            Feature extractor and sequence models will need to be retrained. ' 
            Inference files will be deleted, and feature extractor inference must be re-run.
            If you have not exported predictions to .CSV, make sure you do so now!'''.format(text)
        if not simple_popup_question(self, message):
            return
        if not self.saved:
            if simple_popup_question(self, 'You have unsaved changes. Do you want to save them first?'):
                self.save()
        projects.add_behavior_to_project(os.path.join(self.cfg.project.path, 'project_config.yaml'), text)
        behaviors = OmegaConf.to_container(self.cfg.project.class_names)
        behaviors.append(text)
        self.cfg.project.class_names = behaviors
        # self.project_config['project']['class_names'].append(text)
        # self.project_config = projects.load_config(
        #     os.path.join(self.project_config['project']['path'], 'project_config.yaml'))
        # behaviors = self.project_config['project']['class_names']

        self.import_labelfile(self.labelfile)
        self.thresholds = None
        self.outputfile = None
        self.import_outputfile(self.outputfile)

        self.update()

    def remove_class(self):
        text, ok = QInputDialog.getText(self, 'RemoveBehaviorDialog', 'Enter behavior name: ')
        if len(text) == 0:
            log.warning('No behavior entered')
            ok = False
        if not ok:
            return

        if text not in self.cfg.project.class_names:
            log.warning('This behavior is not in the list...')
            return
        if text == 'background':
            raise ValueError('Cannot remove background class.')

        message = '''Are you sure you want to remove behavior {}? 
            All previous labels will have be DELETED!
            Feature extractor and sequence models will need to be retrained. ' 
            Inference files will be deleted, and feature extractor inference must be re-run.
            If you have not exported predictions to .CSV, make sure you do so now!'''.format(text)
        if not simple_popup_question(self, message):
            return
        projects.remove_behavior_from_project(os.path.join(self.cfg.project.path, 'project_config.yaml'), text)

        behaviors = OmegaConf.to_container(self.cfg.project.class_names)
        behaviors.remove(text)
        self.cfg.project.class_names = behaviors

        # self.project_config = projects.load_config(os.path.join(self.project_config['project']['path'],
        #                                                         'project_config.yaml'))
        # self.project_config['class_names'].remove(text)
        # self.project_config['class_names'].append(text)
        self.outputfile = None
        self.import_labelfile(self.labelfile)
        self.import_outputfile(self.outputfile)

        self.update()

    def finalize(self):
        if not hasattr(self, 'cfg'):
            raise ValueError('cant finalize labels without starting or loading a DEG project')
        message = 'Are you sure you want to continue? All non-labeled frames will be labeled as *background*.\n' \
                  'This is not reversible.'
        if not simple_popup_question(self, message):
            return

        if not self.saved:
            self.save()
            # if simple_popup_question(
            #         self, 'You have unsaved changes. You must save before labels can be finalized. '
            #         'Do you want to save?'):
            #     self.save()
            # else:
            #     return
        log.info('finalizing labels for file {}'.format(self.videofile))
        fname, _ = os.path.splitext(self.videofile)
        label_fname = fname + '_labels.csv'
        if not os.path.isfile(label_fname):
            label = self.ui.labels.label.array
            df = pd.DataFrame(label, columns=self.cfg.project.class_names)
        else:
            df = pd.read_csv(label_fname, index_col=0)
        array = df.values
        array[array == -1] = 0
        background_frames = np.logical_not(np.any(array[:, 1:], axis=1))
        array[background_frames, 0] = 1
        new_df = pd.DataFrame(data=array, columns=list(df.columns))
        new_df.to_csv(label_fname)

        self.ui.labels.label.changed[:] = 1
        self.ui.labels.label.array = array
        self.ui.labels.label.recreate_label_image()
        self.user_did_something()
        self.unfinalized_idx += 1
        if self.unfinalized_idx >= len(self.unfinalized):
            raise ValueError('no more videos to label! you can still use the GUI to browse or fix labels')
        self.initialize_video(self.unfinalized[self.unfinalized_idx]['rgb'])
        

    def save(self):
        if self.saved:
            # do nothing
            return
        log.info('saving...')
        df = self._make_dataframe()
        fname, _ = os.path.splitext(self.videofile)
        label_fname = fname + '_labels.csv'
        df.to_csv(label_fname)
        projects.add_file_to_subdir(label_fname, os.path.dirname(self.videofile))
        # self.save_to_hdf5()
        self.saved = True

    def import_labelfile(self, labelfile: Union[str, os.PathLike]):
        if labelfile is None:
            self.initialize_label()
        assert (os.path.isfile(labelfile))
        df = pd.read_csv(labelfile, index_col=0)
        array = df.values
        self.initialize_label(label_array=array)

    def import_external_labels(self):
        if self.data_path is not None:
            data_dir = self.data_path
        else:
            raise ValueError('create or load a DEG project before importing video')

        options = QFileDialog.Options()
        filestring = 'Label file (*.csv)'
        labelfile, _ = QFileDialog.getOpenFileName(self,
                                                   "Click on labels to import",
                                                   data_dir,
                                                   filestring,
                                                   options=options)
        if projects.is_deg_file(labelfile):
            raise ValueError('Don' 't use this to open labels: use to import non-DeepEthogram labels')
        filestring = 'VideoReader files (*.h5 *.avi *.mp4)'
        videofile, _ = QFileDialog.getOpenFileName(self,
                                                   "Click on corresponding video file",
                                                   data_dir,
                                                   filestring,
                                                   options=options)
        if not projects.is_deg_file(videofile):
            raise ValueError('Please select the already-imported video file that corresponds to the label file.')
        label_dst = projects.add_label_to_project(labelfile, videofile)

        self.import_labelfile(label_dst)

    def import_outputfile(self, outputfile: Union[str, os.PathLike], latent_name=None, first_time: bool = False):

        if outputfile is None:
            self.initialize_prediction()
            return
        try:
            outputs = projects.import_outputfile(self.cfg.project.path,
                                                 outputfile,
                                                 class_names=OmegaConf.to_container(self.cfg.project.class_names),
                                                 latent_name=latent_name)
        except ValueError as e:
            log.exception(e)
            print('If you got a broadcasting error: did you add or remove behaviors and not re-train?')
            self.initialize_prediction()
            return
        probabilities, thresholds, latent_name, keys = outputs

        self.postprocessor = get_postprocessor_from_cfg(self.cfg, thresholds)
        estimated_labels = self.postprocessor(probabilities)
        # this is for visualization purposes only
        probabilities[estimated_labels == 1] = 1

        self.probabilities = probabilities
        self.estimated_labels = estimated_labels
        self.thresholds = thresholds

        opacity = estimated_labels.copy().T.astype(float)
        log.debug('estimated labels: {}'.format(opacity))
        opacity[opacity == 0] = self.cfg.prediction_opacity
        log.debug('opacity array: {}'.format(opacity))

        if np.any(probabilities > 1):
            log.warning('Probabilities > 1 found, clamping...')
            probabilities = probabilities.clip(min=0, max=1.0)

        # import pdb
        # pdb.set_trace()x
        # print('probabilities min: {}, max: {}'.format(probabilities.min(), probabilities.max()))

        self.initialize_prediction(prediction_array=probabilities, opacity=opacity)
        self.ui.importPredictions.setEnabled(True)
        self.ui.exportPredictions.setEnabled(True)
        log.info('CHANGING LATENT NAME TO : {}'.format(latent_name))
        self.latent_name = latent_name

        log.debug('keys: {}'.format(keys))
        if first_time:
            self.ui.predictionsCombo.blockSignals(True)
            self.ui.predictionsCombo.clear()
            for key in keys:
                self.ui.predictionsCombo.addItem(key)
            self.ui.predictionsCombo.blockSignals(False)
        if self.ui.predictionsCombo.currentText() != latent_name:
            self.ui.predictionsCombo.setCurrentText(latent_name)
        self.update()
        self.user_did_something()

    def export_predictions(self):
        array = self.estimated_labels
        np.set_printoptions(suppress=True)
        df = pd.DataFrame(data=array, columns=self.cfg.project.class_names)
        print(df)
        print(df.sum(axis=0))
        fname, _ = os.path.splitext(self.videofile)
        prediction_fname = fname + '_predictions.csv'
        df.to_csv(prediction_fname)

    def change_predictions(self, new_text):
        log.debug('change predictions called with text: {}'.format(new_text))
        log.debug('current latent name: {}'.format(self.latent_name))
        if not hasattr(self, 'outputfile') or new_text is None:
            return
        if self.latent_name != new_text:
            log.debug('not equal found: {}, {}'.format(self.latent_name, new_text))
            #
            self.import_outputfile(self.outputfile, latent_name=new_text)
        # log.warning('prediction import not implemented')

    def import_predictions_as_labels(self):
        if not hasattr(self, 'estimated_labels'):
            raise ValueError('Cannot import predictions before an outputfile has been imported.\n'
                             'Run inference on the feature extractors or sequence models first.')
        should_overwrite_all = overwrite_or_not(self)
        log.debug('should_overwrite_all: {}'.format(should_overwrite_all))

        current_label = self.ui.labels.label.array.copy()
        changed = self.ui.labels.label.changed.copy()
        current_predictions = self.estimated_labels
        if not should_overwrite_all:
            rows_to_change = np.where(changed == 0)[0]
        else:
            rows_to_change = np.arange(0, current_label.shape[0])

        # print(current_label[rows_to_change, :])
        current_label[rows_to_change, :] = current_predictions[rows_to_change, :]
        changed[:] = 1
        self.ui.labels.label.array = current_label
        self.ui.labels.label.changed = changed
        self.ui.labels.label.recreate_label_image()
        self.saved = False
        self.save()
        self.user_did_something()
        # print(changed)

    def open_avi_browser(self):
        if self.data_path is not None:
            data_dir = self.data_path
        else:
            raise ValueError('create or load a DEG project before loading video')

        options = QFileDialog.Options()
        filestring = 'VideoReader files (*.h5 *.avi *.mp4 *.png *.jpg *.mov)'
        prompt = "Click on video to open. If a directory full of images, click any image"
        filename, _ = QFileDialog.getOpenFileName(self, prompt, data_dir, filestring, options=options)
        if len(filename) == 0 or not os.path.isfile(filename):
            raise ValueError('Could not open file: {}'.format(filename))
        ext = os.path.splitext(filename)[1]

        if ext in ['.png', '.jpg']:
            filename = os.path.dirname(filename)
            assert os.path.isdir(filename)

        self.initialize_video(filename)
        self.user_did_something()

    def add_multiple_videos(self):
        if self.data_path is not None:
            data_dir = self.data_path
        else:
            raise ValueError('create or load a DEG project before loading video')

        # https://stackoverflow.com/questions/38252419/how-to-get-qfiledialog-to-select-and-return-multiple-folders
        options = QFileDialog.Options()
        filestring = 'VideoReader files (*.h5 *.avi *.mp4 *.png *.jpg *.mov)'
        prompt = "Click on video to open. If a directory full of images, click any image"
        filenames, _ = QFileDialog.getOpenFileNames(self, prompt, data_dir, filestring, options=options)
        if len(filenames) == 0:
            return
        for filename in filenames:
            assert os.path.exists(filename)

        for filename in filenames:
            self.initialize_video(filename)
        # if len(filename) == 0 or not os.path.isfile(filename):
        #     raise ValueError('Could not open file: {}'.format(filename))
        #
        # self.initialize_video(filename)

    def initialize_project(self, directory: Union[str, os.PathLike]):

        if len(directory) == 0:
            return
        filename = os.path.join(directory, 'project_config.yaml')

        if len(filename) == 0 or not os.path.isfile(filename):
            log.error('something wrong with loading yaml file: {}'.format(filename))
            return

        # project_dict = projects.load_config(filename)
        # if not projects.is_config_dict(project_dict):
        #     raise ValueError('Not a properly formatted configuration dictionary! Look at defaults/config.yaml: dict: {}'
        #                      'filename: {}'.format(project_dict, filename))
        # self.project_config = project_dict
        # # self.project_config = projects.convert_config_paths_to_absolute(self.project_config)

        # # for convenience
        # self.data_path = os.path.join(self.project_config['project']['path'],
        #                               self.project_config['project']['data_path'])
        # self.model_path = os.path.join(self.project_config['project']['path'],
        #                                self.project_config['project']['model_path'])

        # overwrite cfg passed at command line now that we know the project path. still includes command line arguments
        self.cfg = configuration.make_config(directory, ['config', 'gui', 'postprocessor'], run_type='gui', model=None)
        log.info('cwd: {}'.format(os.getcwd()))
        self.cfg = projects.convert_config_paths_to_absolute(self.cfg, raise_error_if_pretrained_missing=False)
        log.info('cwd: {}'.format(os.getcwd()))
        self.cfg = projects.setup_run(self.cfg, raise_error_if_pretrained_missing=False)
        log.info('loaded project configuration: {}'.format(OmegaConf.to_yaml(self.cfg)))
        log.info('cwd: {}'.format(os.getcwd()))
        # for convenience
        self.data_path = self.cfg.project.data_path
        self.model_path = self.cfg.project.model_path

        # self.project_config['project']['class_names'] = self.project_config['class_names']
        # load up the last alphabetic record, which if user includes dates in filename, will be the most recent
        # data_path = os.path.join(self.project_config['project']['path'], self.project_config['project']['data_path'])
        records = projects.get_records_from_datadir(self.data_path)
        self.project_loaded_buttons()
        if len(records) == 0:
            return
        if len(self.unfinalized) == 0:
            last_record = list(records.values())[-1]
        else:
            last_record = self.unfinalized[0]
        if last_record['rgb'] is not None:
            self.initialize_video(last_record['rgb'])
        if last_record['label'] is not None:
            self.import_labelfile(last_record['label'])
        # if last_record['output'] is not None:
        #     self.import_outputfile(last_record['output'])

        self.get_trained_models()

    def load_project(self):
        # options = QFileDialog.Options()

        directory = QFileDialog.getExistingDirectory(self, "Open your deepethogram directory (containing project "
                                                     "config)")
        self.initialize_project(directory)

        # pprint.pprint(self.trained_model_dict)

    def get_default_archs(self):
        # TODO: replace this default logic with hydra 1.0
        if 'preset' in self.cfg:
            preset = self.cfg.preset
        else:
            preset = 'deg_f'
        default_archs = projects.load_default('preset/{}'.format(preset))
        seq_default = projects.load_default('model/sequence')
        default_archs['sequence'] = {'arch': seq_default['sequence']['arch']}

        if 'feature_extractor' in self.cfg and self.cfg.feature_extractor.arch is not None:
            default_archs['feature_extractor']['arch'] = self.cfg.feature_extractor.arch
        if 'flow_generator' in self.cfg and self.cfg.flow_generator.arch is not None:
            default_archs['flow_generator']['arch'] = self.cfg.flow_generator.arch
        if 'sequence' in self.cfg and 'arch' in self.cfg.sequence and self.cfg.sequence.arch is not None:
            default_archs['sequence']['arch'] = self.cfg.sequence.arch
        self.default_archs = default_archs
        log.debug('default archs: {}'.format(default_archs))

    def get_trained_models(self):
        trained_models = projects.get_weights_from_model_path(self.model_path)
        self.get_default_archs()
        log.debug('trained models found: {}'.format(trained_models))
        trained_dict = {}

        self.trained_model_dict = trained_dict
        for model, archs in trained_models.items():
            trained_dict[model] = {}

            # for sequence models, we can train with no pre-trained weights
            if model == 'sequence':
                trained_dict[model][''] = None

            arch = self.default_archs[model]['arch']
            if arch not in archs.keys():
                continue
            trained_dict[model]['no pretrained weights'] = None
            for run in trained_models[model][arch]:
                key = os.path.basename(os.path.dirname(run))
                if key == 'lightning_checkpoints':
                    key = os.path.basename(os.path.dirname(os.path.dirname(run)))
                trained_dict[model][key] = run

        log.debug('trained model dict: {}'.format(self.trained_model_dict))
        models = self.trained_model_dict['flow_generator']
        if len(models) > 0:
            self.ui.flowSelector.clear()
            for key in models.keys():
                self.ui.flowSelector.addItem(key)
            self.ui.flowSelector.setCurrentIndex(len(models) - 1)

        models = self.trained_model_dict['feature_extractor']
        if len(models) > 0:
            self.ui.feSelector.clear()
            for key in models:
                self.ui.feSelector.addItem(key)
            self.ui.feSelector.setCurrentIndex(len(models) - 1)

        models = self.trained_model_dict['sequence']
        if len(models) > 0:
            self.ui.sequenceSelector.clear()
            for key in models:
                self.ui.sequenceSelector.addItem(key)
            self.ui.sequenceSelector.setCurrentIndex(len(models) - 1)

        # print(self.get_selected_models())

    def get_selected_models(self, model_type: str = None):
        flow_model = None
        fe_model = None
        seq_model = None

        models = {'flow_generator': flow_model, 'feature_extractor': fe_model, 'sequence': seq_model}

        if not hasattr(self, 'trained_model_dict'):
            if model_type is not None:
                log.warning('No {} weights found. Please download using the link on GitHub: {}'.format(
                    model_type, 'https://github.com/jbohnslav/deepethogram'))

            return models
        log.info(self.trained_model_dict)
        flow_text = self.ui.flowSelector.currentText()
        if flow_text in list(self.trained_model_dict['flow_generator'].keys()):
            models['flow_generator'] = self.trained_model_dict['flow_generator'][flow_text]

        fe_text = self.ui.feSelector.currentText()
        if fe_text in self.trained_model_dict['feature_extractor'].keys():
            models['feature_extractor'] = self.trained_model_dict['feature_extractor'][fe_text]

        seq_text = self.ui.sequenceSelector.currentText()
        if seq_text in self.trained_model_dict['sequence'].keys():
            models['sequence'] = self.trained_model_dict['sequence'][seq_text]
        return models

    def update_frame(self, n):
        self.ui.videoPlayer.videoView.update_frame(n)

    def move_n_frames(self, n):
        if not hasattr(self, 'vid'):
            return
        x = self.ui.videoPlayer.videoView.current_fnum
        self.ui.videoPlayer.videoView.update_frame(x + n)

    def _make_dataframe(self):
        if not hasattr(self, 'cfg'):
            raise ValueError('attempted to save dataframe without initializing or opening a project')
        label = np.copy(self.ui.labels.label.array).astype(np.int16)
        changed = np.copy(self.ui.labels.label.changed).astype(bool)
        n_behaviors = label.shape[1]
        # all rows that have not been changed by the labeler should be labeled -1!
        # this signifies unknown label. If we left it 0, an unlabeled frame would be indistinguishable from a
        # negative example
        null_row = np.ones((n_behaviors,)) * -1
        label[np.logical_not(changed), :] = null_row
        # print(array.shape, len(self.label.behaviors))
        # print(self.label.behaviors)
        df = pd.DataFrame(data=label, columns=self.cfg.project.class_names)
        return df

    def check_saved(self):
        if not hasattr(self.ui, 'labels'):
            return True
        return self.ui.labels.label.saved

    def closeEvent(self, event, *args, **kwargs):
        super(QMainWindow, self).closeEvent(event, *args, **kwargs)
        # https://stackoverflow.com/questions/1414781/prompt-on-exit-in-pyqt-application
        if not self.saved:
            message = 'You have unsaved changes. Are you sure you want to quit?'
            if simple_popup_question(self, message):
                event.accept()
            else:
                event.ignore()
                return

        if hasattr(self, 'training_pipe'):
            message = 'If you quit, training will be stopped. Are you sure you want to quit?'
            if simple_popup_question(self, message):
                event.accept()
            else:
                event.ignore()
                return
        if hasattr(self, 'inference_pipe'):
            message = 'If you quit, inference will be stopped. Are you sure you want to quit?'
            if simple_popup_question(self, message):
                event.accept()
            else:
                event.ignore()
                return

        if hasattr(self, 'vid'):
            self.vid.close()

    @Slot(bool)
    def update_saved(self, has_been_saved: bool):
        # print('label change signal heard')
        self.saved = has_been_saved


def set_style(app):
    # https://www.wenzhaodesign.com/devblog/python-pyside2-simple-dark-theme
    # button from here https://github.com/persepolisdm/persepolis/blob/master/persepolis/gui/palettes.py
    app.setStyle(QtWidgets.QStyleFactory.create("fusion"))

    darktheme = QtGui.QPalette()
    darktheme.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    darktheme.setColor(QtGui.QPalette.WindowText, QtGui.QColor(222, 222, 222))
    darktheme.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    darktheme.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(222, 222, 222))
    darktheme.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(222, 222, 222))
    # darktheme.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(0, 222, 0))
    darktheme.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(222, 222, 222))
    darktheme.setColor(QtGui.QPalette.Highlight, QtGui.QColor(45, 45, 45))
    darktheme.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Light, QtGui.QColor(60, 60, 60))
    darktheme.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, QtGui.QColor(50, 50, 50))
    darktheme.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(111, 111, 111))
    darktheme.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(122, 118, 113))
    darktheme.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor(122, 118, 113))
    darktheme.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Base, QtGui.QColor(32, 32, 32))

    # darktheme.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 255, 0))
    # darktheme.setColor(QtGui.QPalette.
    # darktheme.setColor(QtGui.QPalette.Background, QtGui.QColor(255,0,0))
    # print(dir(QtGui.QPalette))
    # Define the pallet color
    # Then set the pallet color

    app.setPalette(darktheme)
    return app


def setup_gui_cfg():
    config_list = ['config', 'gui']
    run_type = 'gui'
    model = None

    project_path = projects.get_project_path_from_cl(sys.argv, error_if_not_found=False)
    if project_path is not None:
        cfg = configuration.make_config(project_path, config_list, run_type, model, use_command_line=True)
    else:
        command_line_cfg = OmegaConf.from_cli()
        if 'preset' in command_line_cfg:
            config_list.append('preset/' + command_line_cfg.preset)
        cfgs = [configuration.load_config_by_name(i) for i in config_list]
        cfg = OmegaConf.merge(*cfgs, command_line_cfg)
    try:
        cfg = projects.setup_run(cfg)
    except Exception:
        pass

    # OmegaConf.set_struct(cfg, False)

    log.info('CWD: {}'.format(os.getcwd()))
    log.info('Configuration used: {}'.format(OmegaConf.to_yaml(cfg)))
    return cfg


def run() -> None:

    app = QtWidgets.QApplication(sys.argv)
    app = set_style(app)

    cfg = setup_gui_cfg()

    window = MainWindow(cfg)
    window.resize(1024, 768)
    window.show()

    sys.exit(app.exec_())


# this function is required to allow automatic detection of the module name when running
# from a binary script.
# it should be called from the executable script and not the hydra.main() function directly.
def entry() -> None:
    run()


if __name__ == '__main__':
    run()
