import logging
import os
import subprocess
import time
import warnings
from functools import partial
from typing import Union

import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Signal, Slot

from deepethogram.file_io import VideoReader
# these define the parameters of the deepethogram colormap below
from deepethogram.viz import Mapper

log = logging.getLogger(__name__)


def numpy_to_qpixmap(image: np.ndarray) -> QtGui.QPixmap:
    if image.dtype == np.float:
        image = float_to_uint8(image)
    H, W, C = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    if C == 4:
        format = QtGui.QImage.Format_RGBA8888
    elif C == 3:
        format = QtGui.QImage.Format_RGB888
    else:
        raise ValueError('Aberrant number of channels: {}'.format(C))
    qpixmap = QtGui.QPixmap(QtGui.QImage(image, W, H, image.strides[0], format))
    # print(type(qpixmap))
    return (qpixmap)


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float:
        image = (image * 255).clip(min=0, max=255).astype(np.uint8)
    # print(image)
    return image


def initializer(nframes: int):
    print('initialized with {}'.format(nframes))


class VideoFrame(QtWidgets.QGraphicsView):
    frameNum = Signal(int)
    initialized = Signal(int)
    photoClicked = Signal(QtCore.QPoint)

    def __init__(self, videoFile: Union[str, os.PathLike] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.videoView = QtWidgets.QGraphicsView()

        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        # self.videoView.setScene(self._scene)
        self.setScene(self._scene)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(640, 480))
        # self.setObjectName("videoView")
        self._zoom = 0

        if videoFile is not None:
            self.initialize_video(videoFile)
            self.update()
        self.setStyleSheet("background:transparent;")
        # print(self.palette())

    def initialize_video(self, videofile: Union[str, os.PathLike]):
        if hasattr(self, 'vid'):
            self.vid.close()
            # if hasattr(self.vid, 'cap'):
            #     self.vid.cap.release()
        self.videofile = videofile

        self.vid = VideoReader(videofile)
        # self.frame = next(self.vid)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.initialized.emit(len(self.vid))
        self.update_frame(0)
        self.fitInView()

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if hasattr(self, 'vid'):
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                # https://stackoverflow.com/questions/58965209/zoom-on-mouse-position-qgraphicsview
                view_pos = event.pos()
                scene_pos = self.mapToScene(view_pos)
                self.centerOn(scene_pos)
                self.scale(factor, factor)
                delta = self.mapToScene(view_pos) - self.mapToScene(self.viewport().rect().center())
                self.centerOn(scene_pos - delta)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0
                self.fitInView()

    def update_frame(self, value, force: bool = False):
        # print('updating')
        # print('update to: {}'.format(value))
        # print(self.current_fnum)
        # previous_frame = self.current_fnum
        if not hasattr(self, 'vid'):
            return
        value = int(value)
        if hasattr(self, 'current_fnum'):
            if self.current_fnum == value and not force:
                # print('already there')
                return
        if value < 0:
            # warnings.warn('Desired frame less than 0: {}'.format(value))
            value = 0
        if value >= self.vid.nframes:
            # warnings.warn('Desired frame beyond maximum: {}'.format(self.vid.nframes))
            value = self.vid.nframes - 1

        self.frame = self.vid[value]

        # the frame in the videoreader is the position of the reader. If you've read frame 0, the current reader
        # position is 1. This makes cv2.CAP_PROP_POS_FRAMES match vid.fnum. However, we want to keep track of our
        # currently displayed image, which is fnum - 1
        self.current_fnum = self.vid.fnum - 1
        # print('new fnum: {}'.format(self.current_fnum))
        self.show_image(self.frame)
        self.frameNum.emit(self.current_fnum)

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            # if self.hasPhoto():
            unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
            # print(factor, viewrect, scenerect)
            self.scale(factor, factor)
            self._zoom = 0

    def adjust_aspect_ratio(self):
        if not hasattr(self, 'vid'):
            raise ValueError('Trying to set GraphicsView aspect ratio before video loaded.')
        if not hasattr(self.vid, 'width'):
            self.vid.width, self.vid.height = self.frame.shape[1], self.frame.shape[0]
        video_aspect = self.vid.width / self.vid.height
        H, W = self.height(), self.width()
        new_width = video_aspect * H
        if new_width < W:
            self.setFixedWidth(new_width)
        new_height = W / self.vid.width * self.vid.height
        if new_height < H:
            self.setFixedHeight(new_height)

    def show_image(self, array):
        qpixmap = numpy_to_qpixmap(array)
        self._photo.setPixmap(qpixmap)
        # self.fitInView()
        self.update()
        # self.show()

    def resizeEvent(self, event):
        if hasattr(self, 'vid'):
            pass
            # self.fitInView()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.fitInView()
        return super().mouseDoubleClickEvent(event)


class ScrollbarWithText(QtWidgets.QWidget):
    position = Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.horizontalWidget = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalWidget.sizePolicy().hasHeightForWidth())
        self.horizontalWidget.setSizePolicy(sizePolicy)
        self.horizontalWidget.setMaximumSize(QtCore.QSize(16777215, 25))
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout.setObjectName("horizontalLayout")

        self.horizontalScrollBar = QtWidgets.QScrollBar(self.horizontalWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalScrollBar.sizePolicy().hasHeightForWidth())
        self.horizontalScrollBar.setSizePolicy(sizePolicy)
        self.horizontalScrollBar.setMaximumSize(QtCore.QSize(16777215, 25))
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.horizontalLayout.addWidget(self.horizontalScrollBar)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.horizontalWidget)
        self.plainTextEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plainTextEdit.sizePolicy().hasHeightForWidth())
        self.plainTextEdit.setSizePolicy(sizePolicy)
        self.plainTextEdit.setMaximumSize(QtCore.QSize(100, 25))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plainTextEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.horizontalLayout.addWidget(self.plainTextEdit)
        self.setLayout(self.horizontalLayout)
        # self.ui.plainTextEdit.textChanged.connect
        self.plainTextEdit.textChanged.connect(self.text_change)
        self.horizontalScrollBar.sliderMoved.connect(self.scrollbar_change)
        self.horizontalScrollBar.valueChanged.connect(self.scrollbar_change)

        self.update()
        # self.show()

    def sizeHint(self):
        return QtCore.QSize(480, 25)

    def text_change(self):
        value = self.plainTextEdit.document().toPlainText()
        value = int(value)
        self.position.emit(value)

    def scrollbar_change(self):
        value = self.horizontalScrollBar.value()
        self.position.emit(value)

    @Slot(int)
    def update_state(self, value: int):
        if self.plainTextEdit.document().toPlainText() != '{}'.format(value):
            self.plainTextEdit.setPlainText('{}'.format(value))

        if self.horizontalScrollBar.value() != value:
            self.horizontalScrollBar.setValue(value)

    @Slot(int)
    def initialize_state(self, value: int):
        # print('nframes: ', value)
        self.horizontalScrollBar.setMaximum(value - 1)
        self.horizontalScrollBar.setMinimum(0)
        # self.horizontalScrollBar.sliderMoved.connect(self.scrollbar_change)
        # self.horizontalScrollBar.valueChanged.connect(self.scrollbar_change)
        self.horizontalScrollBar.setValue(0)
        self.plainTextEdit.setPlainText('{}'.format(0))
        # self.plainTextEdit.textChanged.connect(self.text_change)
        # self.update()


class VideoPlayer(QtWidgets.QWidget):
    # added parent here because python-uic, which turns Qt Creator files into python files, always adds the parent
    # widget. so instead of just saying self.videoPlayer = VideoPlayer(), it does
    # self.videoPlayer = VideoPlayer(self.centralWidget)
    # this just means you are required to pass videoFile as a kwarg
    def __init__(self, parent=None, videoFile: Union[str, os.PathLike] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout()

        # initialize both widgets and add it to the vertical layout
        self.videoView = VideoFrame(videoFile)
        layout.addWidget(self.videoView)
        self.scrollbartext = ScrollbarWithText()
        layout.addWidget(self.scrollbartext)

        self.setLayout(layout)

        # if you use the scrollbar or the text box, update the video frame
        # self.scrollbartext.horizontalScrollBar.sliderMoved.connect(self.videoView.update_frame)
        # self.scrollbartext.horizontalScrollBar.valueChanged.connect(self.videoView.update_frame)
        # self.scrollbartext.plainTextEdit.textChanged.connect(self.videoView.update_frame)
        self.scrollbartext.position.connect(self.videoView.update_frame)
        self.scrollbartext.position.connect(self.scrollbartext.update_state)

        # if you move the video by any method, update the frame text
        self.videoView.initialized.connect(self.scrollbartext.initialize_state)
        # self.videoView.initialized.connect(initializer)
        self.videoView.frameNum.connect(self.scrollbartext.update_state)

        # I have to do this here because I think emitting a signal doesn't work from within the widget's constructor
        if hasattr(self.videoView, 'vid'):
            self.videoView.initialized.emit(len(self.videoView.vid))

        self.update()


# class LabelImage(QtWidgets.QScrollArea):
#     def __init__(self, parent=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         layout = QtWidgets.QHBoxLayout()
#         self.widget = QtWidgets.QWidget()
#
#         buttonlayout = QtWidgets.QVBoxLayout()
#         self.labels = []
#         sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
#         sizePolicy.setHorizontalStretch(0)
#         sizePolicy.setVerticalStretch(0)
#         for i in range(100):
#             self.labels.append(QtWidgets.QLabel('testing{}'.format(i)))
#             self.labels[i].setMinimumHeight(25)
#             buttonlayout.addWidget(self.labels[i])
#             # self.labels[i].setLayout(buttonlayout)
#
#         self.widget.setLayout(buttonlayout)
#         self.setWidget(self.widget)
#
#         self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
#         self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#
#         self.update()
#
#     def sizeHint(self):
#         return (QtCore.QSize(720, 250))
# https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python

# start = np.array([232,232,232])


class LabelViewer(QtWidgets.QGraphicsView):
    X = Signal(int)
    saved = Signal(bool)
    just_toggled = Signal(bool)
    num_changed = Signal(int)

    def __init__(self, fixed: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        # self.videoView.setScene(self._scene)
        self.setScene(self._scene)
        color = QtGui.QColor(45, 45, 45)
        self.pen = QtGui.QPen(color, 0)

        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        # self.setAlignment(QtCore.Qt.AlignCenter)
        # self.setStyleSheet("background:transparent;")
        self.initialized = False

        self.fixed = fixed
        if self.fixed:
            self.fixed_settings()

    def initialize(self,
                   n: int = 1,
                   n_timepoints: int = 31,
                   debug: bool = False,
                   colormap: str = 'Reds',
                   unlabeled_alpha: float = 0.1,
                   desired_pixel_size: int = 25,
                   array: np.ndarray = None,
                   fixed: bool = False,
                   opacity: np.ndarray = None):
        if self.initialized:
            raise ValueError('only initialize once!')
        if array is not None:
            # print(array.shape)
            self.n_timepoints = array.shape[0]
            self.n = array.shape[1]
            # if our input array is -1s, assume that this has not been labeled yet
            self.changed = np.any(array != -1, axis=1).astype(np.uint8)
            array[array == -1] = 0
            self.array = array
            self.debug = False
        else:
            self.n = n
            self.array = np.zeros((n_timepoints, self.n), dtype=np.uint8)
            self.changed = np.zeros((n_timepoints,), dtype=np.uint8)
            self.n_timepoints = n_timepoints
            self.debug = debug
        self.shape = self.array.shape

        self.label_toggled = np.array([False for i in range(n)])
        self.desired_pixel_size = desired_pixel_size

        try:
            self.cmap = Mapper(colormap)
        except ValueError as e:
            raise ('Colormap not in matplotlib' 's defaults! {}'.format(colormap))
        if self.debug:
            self.make_debug()

        self.unlabeled_alpha = unlabeled_alpha
        self.opacity = opacity
        self.recreate_label_image()
        pos_colors = self.cmap(np.ones((self.n, 1)) * 255)
        neg_colors = self.cmap(np.zeros((self.n, 1)))
        # print('N: {}'.format(self.n))
        self.pos_color = np.array([pos_colors[i].squeeze() for i in range(self.n)])
        self.neg_color = np.array([neg_colors[i].squeeze() for i in range(self.n)])

        # print('pos, neg: {}, {}'.format(self.pos_color, self.neg_color))

        draw_rect = QtCore.QRectF(0, 0, 1, self.n)
        # print(dir(self.draw_rect))
        self.item_rect = self._scene.addRect(draw_rect, self.pen)

        self.change_view_x(0)
        self.fixed = fixed  # initialization overwrides constructor
        if self.fixed:
            self.fixed_settings()
        self.initialized = True
        self.update()
        self.num_changed.emit(np.sum(self.changed))

    def mousePressEvent(self, event):
        if not self.initialized:
            return
        # print(dir(event))
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()

        # print('X: {} Y: {}'.format(x,y))
        x, y = int(x), int(y)
        value = self.array[x, y]
        if value == 0:
            self._add_behavior([y], x, x)
        else:
            self._add_behavior([y], x + 1, x)
        self.initial_row = y
        self.initial_column = x
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.initialized:
            return
        scene_pos = self.mapToScene(event.pos())
        x, _ = scene_pos.x(), scene_pos.y()
        y = self.initial_row

        # print('X: {} Y: {}'.format(x,y))
        x, y = int(x), int(y)
        # value = self.array[x, y]

        if x > self.initial_column:
            self._add_behavior([y], x, x)
        else:
            self._add_behavior([y], x + 1, x)
        self.last_column = x

        super().mouseMoveEvent(event)

    def change_rectangle(self, rect):
        if not hasattr(self, 'item_rect'):
            return
        self.item_rect.setRect(rect)

    def _fit_label_photo(self):
        if not hasattr(self, 'x'):
            self.x = 0
        if not hasattr(self, 'view_x'):
            self.view_x = 0
        # gets the bounding rectangle (in pixels) for the image of the label array
        geometry = self.geometry()
        # print(geometry)
        widget_width, widget_height = geometry.width(), geometry.height()
        num_pix_high = widget_height / self.desired_pixel_size

        aspect = widget_width / widget_height

        new_height = num_pix_high
        new_width = new_height * aspect
        # print('W: {} H: {}'.format(new_width, new_height))
        rect = QtCore.QRectF(self.view_x, 0, new_width, new_height)

        self.fitInView(rect)
        # self.fitInView(rect, aspectRadioMode=QtCore.Qt.KeepAspectRatio)
        self.view_height = new_height
        self.view_width = new_width
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        self._fit_label_photo()
        return

    @Slot(int)
    def change_view_x(self, x: int):
        if x < 0 or x >= self.n_timepoints:
            # print('return 1')
            return
        if not hasattr(self, 'view_width'):
            self._fit_label_photo()
        if not hasattr(self, 'n'):
            # print('return 2')
            return

        view_x = x - self.view_width // 2

        if view_x < 0:
            # print('desired view x: {} LEFT SIDE'.format(view_x))
            new_x = 0
        elif view_x >= self.n_timepoints:
            # print('desired view x: {} RIGHT SIDE'.format(view_x))
            new_x = self.n_timepoints - 1
        else:
            new_x = view_x

        # new_x = max(view_x, 0)
        # new_x = min(new_x, self.n_timepoints - 1)

        old_x = self.x
        self.view_x = new_x
        self.x = x

        position = QtCore.QPointF(x, 0)
        # print('view width: {}'.format(self.view_width))
        # print('new_x: {}'.format(new_x))
        # print('rec_x: {}'.format(position))
        self.item_rect.setPos(position)
        self.X.emit(self.x)
        rect = QtCore.QRectF(self.view_x, 0, self.view_width, self.view_height)
        # print('View rectangle: {}'.format(rect))
        self.fitInView(rect)

        behaviors = []
        for i, v in enumerate(self.label_toggled):
            if v:
                behaviors.append(i)

        if len(behaviors) > 0:
            self._add_behavior(behaviors, old_x, x)
        # self._fit_label_photo()
        self.update()
        # self.show()

    def fixed_settings(self):
        if not hasattr(self, 'changed'):
            return
        self.changed = np.ones(self.changed.shape)
        self.recreate_label_image()

    def _add_behavior(self, behaviors: Union[int, np.ndarray, list], fstart: int, fend: int):
        # print('adding')
        if self.fixed:
            return
        if not hasattr(self, 'array'):
            return
        n_behaviors = self.image.shape[0]
        if type(behaviors) != np.ndarray:
            behaviors = np.array(behaviors)
        if max(behaviors) > n_behaviors:
            raise ValueError('Not enough behaviors for number: {}'.format(behaviors))
        if fstart < 0:
            raise ValueError('Behavior start frame must be > 0: {}'.format(fstart))
        if fend > self.n_timepoints:
            raise ValueError('Behavior end frame must be < nframes: {}'.format(fend))
        # log.debug('Behaviors: {} fstart: {} fend: {}'.format(behaviors, fstart, fend))
        # go backwards to erase
        if fstart <= fend:
            value = 1
            time_indices = np.arange(fstart, fend + 1)  # want it to be
            color = self.pos_color
            # print('value = 1')
        elif fstart - fend == 1:
            value = 0
            time_indices = np.array([fend, fstart])
            color = self.neg_color
        else:
            # print('value = 0')
            value = 0
            time_indices = np.arange(fstart, fend, -1)
            color = self.neg_color

        # log.debug('time indices: {} value: {}'.format(time_indices, value))

        # handle background specifically
        if len(behaviors) == 1 and behaviors[0] == 0:
            # print('0')
            self.array[time_indices, 0] = 1
            self.array[time_indices, 1:] = 0
            # print('l shape: {}'.format(self.image[1:, time_indices, :].shape))
            # print('r_shape: {}'.format(np.tile(self.neg_color[1:], [1, len(time_indices), 1]).shape))
            self.image[0, time_indices, :] = self.pos_color[0]
            self.image[1:, time_indices, :] = np.dstack([self.neg_color[1:] for _ in range(len(time_indices))
                                                        ]).swapaxes(1, 2)
        else:
            xv, yv = np.meshgrid(time_indices, behaviors, indexing='ij')
            xv = xv.flatten()
            yv = yv.flatten()
            # log.debug('xv: {} yv: {}'.format(xv, yv))
            # print('yv: {}'.format(yv))
            self.array[xv, yv] = value
            # change color
            self.image[yv, xv, :] = color[yv]
            # if there are any changes to rows 1+, make sure background is false
            self.array[time_indices, 0] = np.logical_not(np.any(self.array[time_indices, 1:], axis=1))
            # remap the color for the background column just in case
            self.image[0, time_indices, :] = self.cmap(self.array[time_indices, 0:1].T * 255).squeeze()
        # mapped = self.cmap(self.array[time_indices, 0] * 255)
        # print('mapped in add behavior: {}'.format(mapped.shape))
        # self.image[0, time_indices, :] = mapped
        # print(self.label.image[0,time_indices])
        # change opacity
        self.image[:, time_indices, 3] = 1

        self.changed[time_indices] = 1
        # change opacity
        # self.label.image[:, indices, 3] = 1
        self.saved.emit(False)
        # self.label.image = im
        self.update_image()
        self.num_changed.emit(self.changed.sum())

    def change_view_dx(self, dx: int):
        self.change_view_x(self.x + dx)

    def _array_to_image(self, array: np.ndarray, alpha: Union[float, int, np.ndarray] = None):
        image = self.cmap(array.T * 255)
        image[..., 3] = alpha
        return (image)

    def _add_row(self):
        self.array = np.concatenate((self.array, np.zeros((self.array.shape[0], 1), dtype=self.array.dtype)), axis=1)
        alpha_vector = self.image[0, :, 3:4]

        alpha_array = np.tile(alpha_vector, (1, self.array.shape[1]))
        self.image = self._array_to_image(self.array, alpha_array.T)
        self.n += 1
        self.label_toggled = np.append(self.label_toggled, [False])
        rect = QtCore.QRectF(self.x, 0, 1, self.n)
        self.change_rectangle(rect)
        self._fit_label_photo()

    def _change_n_timepoints(self, n_timepoints: int):
        warnings.warn('Changing number of timepoints will erase any labels!')
        self.array = np.zeros((n_timepoints, self.n), dtype=np.uint8)
        self.changed = np.zeros((n_timepoints,), dtype=np.uint8)
        self.n_timepoints = n_timepoints
        self.shape = self.array.shape
        self.image = self._array_to_image(self.array, alpha=self.unlabeled_alpha)

    def make_debug(self, num_rows: int = 15000):
        print('debug')
        assert (hasattr(self, 'array'))
        rows, cols = self.shape
        # print(rows, cols)
        # behav = 0
        for i in range(rows):
            behav = (i % cols)
            self.array[i, behav] = 1
        # self.array = self.array[:num_rows,:]
        # print(self.array)

    def calculate_background_class(self, array: np.ndarray):
        array[:, 0] = np.logical_not(np.any(array[:, 1:], axis=1))
        return (array)

    def update_background_class(self):
        # import pdb
        # pdb.set_trace()
        self.array = self.calculate_background_class(self.array)

    def update_image(self):
        qpixmap = numpy_to_qpixmap(self.image)
        self.qpixmap = qpixmap
        self._photo.setPixmap(self.qpixmap)
        self.update()

    def recreate_label_image(self):
        # print('array input shape, will be transposed: {}'.format(self.array.shape))
        self.image = self.cmap(self.array.T * 255)
        if self.opacity is None:
            opacity = np.ones((self.image.shape[0], self.image.shape[1])) * self.unlabeled_alpha
            opacity[:, np.where(self.changed)[0]] = 1
        else:
            opacity = self.opacity.copy()
        # print('image: {}'.format(self.image))
        # print('image shape in recreate label image: {}'.format(self.image.shape))
        # print('opacity: {}'.format(opacity))
        # print('opacity shape in recreate label image: {}'.format(opacity.shape))

        # print('chang: {}'.format(self.changed.shape))

        self.image[..., 3] = opacity
        self.update_image()

    @Slot(int)
    def toggle_behavior(self, index: int):
        if not hasattr(self, 'array') or self.array is None or self.fixed:
            return
        n_behaviors = self.image.shape[0]
        if index > n_behaviors:
            raise ValueError('Not enough behaviors for number: {}'.format(index))
        if index < 0:
            raise ValueError('Behavior index cannot be below 0')
        self.label_toggled[index] = ~self.label_toggled[index]
        if self.label_toggled[index]:
            # if background is selected, deselect all others
            if index == 0:
                self.label_toggled[1:] = False
                self.array[self.x, 1:] = 0

            self.array[self.x, index] = 1
            self.changed[self.x] = 1

        self.update_background_class()
        self.recreate_label_image()
        self.change_view_x(self.x)
        # print(self.changed)
        self.just_toggled.emit(index)
        self.update()


class LabelButtons(QtWidgets.QWidget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reset()

    def reset(self):
        self.layout = None
        self.buttons = None
        self.behaviors = None
        self.enabled = None
        self.minimum_height = None

    def initialize(self,
                   behaviors: Union[list, np.ndarray] = ['background'],
                   enabled: bool = True,
                   minimum_height: int = 25):
        assert (len(behaviors) > 0)
        layout = QtWidgets.QVBoxLayout()
        self.buttons = []
        self.behaviors = behaviors
        self.enabled = enabled
        self.minimum_height = minimum_height

        for i, behavior in enumerate(behaviors):
            # not sure if I need the str call but I don't want a weird single-element numpy object
            behavior = str(behavior)
            button = self._make_button(behavior, i)
            self.buttons.append(button)
            layout.addWidget(button, 0, alignment=QtCore.Qt.AlignTop)
        layout.setMargin(0)
        layout.setSpacing(0)
        self.layout = layout
        self.setLayout(self.layout)

    def _make_button(self, behavior: str, index: int):
        string = str(behavior)
        if index < 10:
            string = '[{:01d}] '.format(index) + string
        button = QtWidgets.QPushButton(string, parent=self)
        button.setEnabled(self.enabled)
        button.setMinimumHeight(self.minimum_height)
        button.setCheckable(True)
        button.setStyleSheet("QPushButton { text-align: left; }"
                             "QPushButton:checked { background-color: rgb(30, 30, 30)}")
        return button

    def add_behavior(self, behavior: str):
        if behavior in self.behaviors:
            warnings.warn('behavior {} already in list'.format(behavior))
        else:
            self.behaviors.append(behavior)
        button = self._make_button(behavior, len(self.behaviors))
        self.buttons.append(button)
        self.layout.addWidget(button, 0, alignment=QtCore.Qt.AlignTop)
        self.update()

    def fix(self):
        for button in self.buttons:
            button.setEnabled(False)


class LabelImg(QtWidgets.QScrollArea):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        self.setSizePolicy(sizePolicy)
        self.setMaximumSize(QtCore.QSize(16777215, 200))

        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.reset()

    def reset(self):
        self.label = None
        self.behaviors = None
        self.n = None
        self.buttons = None
        self.toggle_shortcuts = None
        self.widget = None
        self.layout = None

    def update_buttons(self):
        if self.label is None or self.buttons is None:
            return
        toggled = self.label.label_toggled
        for toggle, button in zip(toggled, self.buttons.buttons):
            if toggle != button.isChecked():
                button.setChecked(toggle)
        self.update()

    def initialize(self,
                   behaviors: Union[list, np.ndarray] = ['background'],
                   n_timepoints: int = 31,
                   debug: bool = False,
                   colormap: str = 'Reds',
                   unlabeled_alpha: float = 0.1,
                   desired_pixel_size: int = 25,
                   array: np.ndarray = None,
                   fixed: bool = False,
                   opacity: np.ndarray = None):

        layout = QtWidgets.QHBoxLayout()
        # assert (n == len(behaviors))
        assert (behaviors[0] == 'background')

        self.label = LabelViewer()
        # print(behaviors)
        self.behaviors = behaviors
        self.n = len(self.behaviors)
        self.label.initialize(len(self.behaviors),
                              n_timepoints,
                              debug,
                              colormap,
                              unlabeled_alpha,
                              desired_pixel_size,
                              array,
                              fixed,
                              opacity=opacity)
        self.buttons = LabelButtons()
        enabled = not fixed
        self.buttons.initialize(self.behaviors, enabled, desired_pixel_size)
        if not fixed:
            tmp_buttons = []
            for i, button in enumerate(self.buttons.buttons):
                button.clicked.connect(partial(self.label.toggle_behavior, i))
                tmp_buttons.append(button)
                # self.toggle_shortcuts[i].activated.connect(partial(self.label.toggle_behavior, i))
            self.label.just_toggled.connect(self.update_buttons)
        self.setMinimumHeight(desired_pixel_size)

        # this syntax figured out from here
        # https://www.learnpyqt.com/courses/adanced-ui-features/qscrollarea/
        self.widget = QtWidgets.QWidget()
        layout.addWidget(self.buttons, alignment=QtCore.Qt.AlignTop)
        layout.addWidget(self.label, alignment=QtCore.Qt.AlignTop)
        self.widget.setLayout(layout)
        self.setWidget(self.widget)

        self.update()

    def add_behavior(self, behavior: str):
        print('1: ', self.behaviors, behavior)
        if behavior in self.behaviors:
            warnings.warn('behavior {} already in list'.format(behavior))
        # add a button
        self.buttons.add_behavior(behavior)
        print('2: {}'.format(self.behaviors))
        print('2 buttons: {}'.format(self.buttons.behaviors))
        # add to our list of behaviors
        # self.behaviors.append(behavior)
        print('3: {}'.format(self.behaviors))
        # hook up button to toggling behavior
        i = len(self.behaviors) - 1
        print(self.behaviors)
        print(len(self.behaviors))
        print(len(self.buttons.buttons))
        self.buttons.buttons[i].clicked.connect(partial(self.label.toggle_behavior, i))

        self.label._add_row()
        if i < 10:
            self.toggle_shortcuts.append(QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), self))
        self.toggle_shortcuts[i].activated.connect(self.buttons.buttons[i].click)


class ListenForPipeCompletion(QtCore.QThread):
    has_finished = Signal(bool)

    def __init__(self, pipe):
        QtCore.QThread.__init__(self)
        # super().__init__(self)
        self.pipe = pipe

    def __del__(self):
        self.should_continue = False

    def run(self):
        while self.should_continue:
            time.sleep(1)
            if self.pipe.poll() is None:
                pass
                # print('still running...')
            else:
                self.has_finished.emit(True)
                break


class SubprocessChainer(QtCore.QThread):

    def __init__(self, calls: list):
        QtCore.QThread.__init__(self)
        for call in calls:
            assert type(call) == list
        self.calls = calls
        self.should_continue = True

    def stop(self):
        self.should_continue = False
        # self.pipe.terminate()

    def run(self):
        for call in self.calls:
            if self.should_continue:
                self.pipe = subprocess.Popen(call)
                while True:
                    time.sleep(1)
                    if self.pipe.poll() is not None or not self.should_continue:
                        self.pipe.terminate()
                        self.pipe.wait()
                        break


# def chained_subprocess_calls(calls: list) -> None:
#     def _run(calls):
#         for call in calls:
#             assert type(call) == list
#
#         for call in calls:
#             print(call)
#             pipe = subprocess.run(call, shell=True)
#     thread = threading.Thread(target=_run, args=(calls,))
#     thread.start()
#     return thread


class UnclickButtonOnPipeCompletion(QtCore.QThread):

    def __init__(self, button, pipe):
        QtCore.QThread.__init__(self)
        # super().__init__(self)
        self.button = button
        self.pipe = pipe
        self.should_continue = True
        self.has_been_clicked = False
        self.button.clicked.connect(self.get_click)

    def __del__(self):
        self.should_continue = False

    @Slot(bool)
    def get_click(self, value):
        print('clicked')
        self.has_been_clicked = True

    def run(self):
        while self.should_continue:
            time.sleep(1)
            if self.pipe.poll() is None:
                pass
                # print('still running...')
            else:
                if not self.has_been_clicked:
                    # print('ischecked: ', self.button.isChecked())
                    if self.button.isChecked():
                        # print('listener clicking button')
                        self.button.click()
                break


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.label = LabelImg(self)
        self.label.initialize(behaviors=['background', 'itch', 'lick', 'scratch', 'shit', 'fuck', 'ass', 'bitch'],
                              n_timepoints=500,
                              debug=True,
                              fixed=False)

        # self.label = LabelViewer()
        # self.label.initialize(n=4, n_timepoints=40, debug=True, fixed=True)
        # # self.labelImg = DebuggingDrawing()
        #
        next_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Right'), self)
        next_shortcut.activated.connect(partial(self.label.label.change_view_dx, 1))
        # next_shortcut.activated.connect(partial(self.label.change_view_dx, 1))
        back_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Left'), self)
        back_shortcut.activated.connect(partial(self.label.label.change_view_dx, -1))
        #
        # if hasattr(self, 'label'):
        #     n = self.label.n
        # else:
        #     n = 1
        # self.toggle_shortcuts = []
        # for i in range(n):
        #     self.toggle_shortcuts.append(QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), self))
        #     self.toggle_shortcuts[i].activated.connect(partial(self.label.toggle_behavior, i))

        # self.buttons = LabelButtons(behaviors = ['background', 'itch', 'scratch', 'poop'])

        # back_shortcut.activated.connect(partial(self.labelImg.move_rect, -1))

        # self.labelImg.make_debug(10)
        self.setCentralWidget(self.label)

        self.setMaximumHeight(480)
        self.update()

    def sizeHint(self):
        return (QtCore.QSize(600, 600))


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    # volume = VideoPlayer(r'C:\DATA\mouse_reach_processed\M134_20141203_v001.h5')
    testing = LabelImg()
    testing.initialize(behaviors=['background', 'a', 'b', 'c', 'd', 'e'], n_timepoints=15000, debug=True)
    # testing = ShouldRunInference(['M134_20141203_v001',
    #                               'M134_20141203_v002',
    #                               'M134_20141203_v004'],
    #                              [True, True, False])
    # testing = MainWindow()
    # testing.setMaximumHeight(250)
    testing.update()
    testing.show()
    app.exec_()
