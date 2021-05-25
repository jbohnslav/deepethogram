import pathlib
import warnings

from PySide2 import QtCore, QtWidgets


def simple_popup_question(parent, message: str):
    # message = 'You have unsaved changes. Are you sure you want to quit?'
    reply = QtWidgets.QMessageBox.question(parent, 'Message',
                                           message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    return reply == QtWidgets.QMessageBox.Yes

# https://stackoverflow.com/questions/15682665/how-to-add-custom-button-to-a-qmessagebox-in-pyqt4

def overwrite_or_not(parent):
    msgBox = QtWidgets.QMessageBox(parent)
    msgBox.setIcon(QtWidgets.QMessageBox.Question)
    msgBox.setText('Do you want to overwrite your labels with these predictions, or only import the predictions'
                   ' for frames you haven''t labeled?')
    overwrite = msgBox.addButton('Overwrite', QtWidgets.QMessageBox.YesRole)
    unlabeled = msgBox.addButton('Only import unlabeled', QtWidgets.QMessageBox.NoRole)
    msgBox.exec_()
    if msgBox.clickedButton() is overwrite:
        return True
    elif msgBox.clickedButton() is unlabeled:
        return False
    else:
        return

class OverwriteOrNot(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        msgBox = QtWidgets.QMessageBox()
        msgBox.setText('Do you want to overwrite your labels with these predictions, or only import the predictions'
                       ' for frames you haven''t labeled?')
        msgBox.addButton(QtWidgets.QPushButton('Overwrite'), QtWidgets.QMessageBox.YesRole)
        msgBox.addButton(QtWidgets.QPushButton('Only import unlabeled'), QtWidgets.QMessageBox.NoRole)
        msgBox.exec_()


class CreateProject(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        string = 'Pick directory for your project. SHOULD BE ON YOUR FASTEST HARD DRIVE. VIDEOS WILL BE COPIED HERE'
        project_directory = QtWidgets.QFileDialog.getExistingDirectory(self, string)
        if len(project_directory) == 0:
            warnings.warn('Please choose a directory')
            return
        project_directory = str(pathlib.Path(project_directory).resolve())
        self.project_directory = project_directory

        self.project_name_default = 'Project Name'
        self.project_box = QtWidgets.QLineEdit(self.project_name_default)
        self.label_default_string = 'Name of person labeling'
        self.labeler_box = QtWidgets.QLineEdit(self.label_default_string)
        # self.labeler_box.
        self.behavior_default_string = 'List of behaviors, e.g. \"walk,scratch,itch\". Do not include none,other,' \
                                       'background,etc '
        self.behaviors_box = QtWidgets.QLineEdit(self.behavior_default_string)
        # self.finish_button = QPushButton('Ok')
        # self.cancel_button = QPushButton('Cancel')
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QtWidgets.QFormLayout()
        layout.addRow(self.project_box)
        layout.addRow(self.labeler_box)
        layout.addRow(self.behaviors_box)
        # layout.addRow(hbox)
        layout.addWidget(button_box)
        self.setLayout(layout)
        # win = QtWidgets.QWidget()
        self.setWindowTitle('Create project')
        self.resize(800, 400)
        self.show()


# modified from https://pythonspot.com/pyqt5-form-layout/
class ShouldRunInference(QtWidgets.QDialog):

    def __init__(self, record_keys: list, should_start_checked: list):
        super(ShouldRunInference, self).__init__()

        self.createFormGroupBox(record_keys, should_start_checked)

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.addWidget(self.scrollArea)
        mainLayout.addWidget(buttonBox, alignment=QtCore.Qt.AlignLeft)
        self.setLayout(mainLayout)

        self.setWindowTitle("Select videos to run inference on")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        self.scrollArea.setSizePolicy(sizePolicy)
        # self.scrollArea.setGeometry(QtCore.QSize(400, 25))
        self.scrollArea.setMaximumSize(QtCore.QSize(400, 16777215))

    def createFormGroupBox(self, record_keys: list, should_start_checked: list):
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.buttons = []
        # make a container widget
        self.scrollWidget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        for row, (record, check) in enumerate(zip(record_keys, should_start_checked)):
            button = QtWidgets.QCheckBox(self)
            button.setChecked(check)
            text = QtWidgets.QLabel(record + ':')
            layout.addWidget(text, row, 0)
            layout.addWidget(button, row, 1)
            self.buttons.append(button)
        # put this layout in the container widget
        self.scrollWidget.setLayout(layout)
        # set the scroll area's widget to be the container
        self.scrollArea.setWidget(self.scrollWidget)


    def get_outputs(self):
        if not hasattr(self, 'buttons'):
            return None
        answers = []
        for button in self.buttons:
            answers.append(button.isChecked())
        return answers


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    num = 50
    form = ShouldRunInference(['M134_20141203_v001',
                               'M134_20141203_v002',
                               'M134_20141203_v004']*num,
                              [True, True, False]*num)
    ret = form.exec_()
    if ret:
        print(form.get_outputs())
    # ret = app.exec_()
    # print(ret)
