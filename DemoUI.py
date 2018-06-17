from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from FeatureSearch.Analyzer import *
from FeatureSearch.Classifier import *

import sys


class Ui_FeatureSearchDemo(QMainWindow):
    def setupUi(self, FeatureSearchDemo):
        self.currentDataset = create_dataset('9Files_largescale_onlyCPP_2018-06-15_21_47.arff')
        FeatureSearchDemo.setObjectName("FeatureSearchDemo")
        FeatureSearchDemo.resize(640, 553)
        self.centralwidget = QtWidgets.QWidget(FeatureSearchDemo)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 1, 2, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 0, 1, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 1, 0, 1, 1)
        self.textBrowserCodeViewer = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowserCodeViewer.setEnabled(True)
        self.textBrowserCodeViewer.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.textBrowserCodeViewer.setObjectName("textBrowserCodeViewer")
        font = QtGui.QFont()
        font.setFamily("Lucida Console")
        self.textBrowserCodeViewer.setFont(font)
        self.gridLayout.addWidget(self.textBrowserCodeViewer, 2, 2, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verifiableCodeEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.verifiableCodeEdit.setFont(font)
        self.verifiableCodeEdit.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.verifiableCodeEdit.setObjectName("verifiableCodeEdit")
        self.verticalLayout.addWidget(self.verifiableCodeEdit)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButtonImport = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonImport.setObjectName("pushButtonImport")
        self.horizontalLayout.addWidget(self.pushButtonImport)
        self.comboBoxClassifierSelect = QtWidgets.QComboBox(self.centralwidget)
        self.comboBoxClassifierSelect.setObjectName("comboBoxClassifierSelect")
        self.horizontalLayout.addWidget(self.comboBoxClassifierSelect)
        self.pushButtonVerify = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonVerify.setObjectName("pushButtonVerify")
        self.horizontalLayout.addWidget(self.pushButtonVerify)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.userList = QtWidgets.QListWidget(self.centralwidget)
        self.userList.setObjectName("userList")
        self.verticalLayout_2.addWidget(self.userList)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonDatasetChange = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonDatasetChange.setObjectName("pushButtonDatasetChange")
        self.horizontalLayout_2.addWidget(self.pushButtonDatasetChange)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 2, 1, 1, 1)
        self.codeFileList = QtWidgets.QListWidget(self.centralwidget)
        self.codeFileList.setObjectName("codeFileList")
        self.gridLayout.addWidget(self.codeFileList, 2, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 3)
        self.gridLayout.setColumnStretch(2, 8)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        FeatureSearchDemo.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(FeatureSearchDemo)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        FeatureSearchDemo.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(FeatureSearchDemo)
        self.statusbar.setObjectName("statusbar")
        FeatureSearchDemo.setStatusBar(self.statusbar)
        self.actionOpen_Dataset_Folder = QtWidgets.QAction(FeatureSearchDemo)
        self.actionOpen_Dataset_Folder.setObjectName("actionOpen_Dataset_Folder")
        self.actionCreate_Dataset_From_Folder = QtWidgets.QAction(FeatureSearchDemo)
        self.actionCreate_Dataset_From_Folder.setObjectName("actionCreate_Dataset_From_Folder")
        self.actionQuit = QtWidgets.QAction(FeatureSearchDemo)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionOpen_Dataset_Folder)
        self.menuFile.addAction(self.actionCreate_Dataset_From_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(FeatureSearchDemo)
        self.actionQuit.triggered.connect(FeatureSearchDemo.close)
        QtCore.QMetaObject.connectSlotsByName(FeatureSearchDemo)

    def retranslateUi(self, FeatureSearchDemo):
        _translate = QtCore.QCoreApplication.translate
        FeatureSearchDemo.setWindowTitle(_translate("FeatureSearchDemo", "Feature Search Demonstration"))
        self.verifiableCodeEdit.setPlaceholderText(_translate("FeatureSearchDemo", "Type your code here..."))
        self.textBrowserCodeViewer.setTabStopWidth(28)
        self.verifiableCodeEdit.setTabStopWidth(28)
        self.comboBoxClassifierSelect.addItem("Logistic Regression")
        self.comboBoxClassifierSelect.addItem("Linear Discriminant Analysis")
        self.comboBoxClassifierSelect.addItem("K-Nearest Neighbour")
        self.comboBoxClassifierSelect.addItem("Decision Tree")
        self.comboBoxClassifierSelect.addItem("Naive-Bayes")
        self.comboBoxClassifierSelect.addItem("Support Vector Machine")
        self.comboBoxClassifierSelect.addItem("Random Forest")
        self.pushButtonImport.setText(_translate("FeatureSearchDemo", "Import from File..."))
        self.pushButtonImport.setShortcut(_translate("FeatureSearchDemo", "I"))
        self.pushButtonVerify.setText(_translate("FeatureSearchDemo", "Verify"))
        self.pushButtonVerify.setShortcut(_translate("FeatureSearchDemo", "V"))
        self.pushButtonDatasetChange.setText(_translate("FeatureSearchDemo", "Change User Directory"))
        self.pushButtonDatasetChange.setShortcut(_translate("FeatureSearchDemo", "C"))
        self.menuFile.setTitle(_translate("FeatureSearchDemo", "File"))
        self.actionOpen_Dataset_Folder.setText(_translate("FeatureSearchDemo", "Change Active Dataset..."))
        self.actionCreate_Dataset_From_Folder.setText(_translate("FeatureSearchDemo", "Create Dataset From Folder..."))
        self.actionQuit.setText(_translate("FeatureSearchDemo", "Quit"))
        self.actionQuit.setShortcut(_translate("FeatureSearchDemo", "Alt+X"))
        self.actionOpen_Dataset_Folder.triggered.connect(self.setDataset)
        self.actionCreate_Dataset_From_Folder.triggered.connect(self.createDataset)
        self.pushButtonDatasetChange.clicked.connect(self.setDataFolder)
        self.pushButtonImport.clicked.connect(self.importCodeFromFile)
        self.pushButtonVerify.clicked.connect(self.verifyCode)
        self.userList.itemClicked.connect(self.showUserFiles)
        self.codeFileList.itemClicked.connect(self.showCode)
        self.comboBoxClassifierSelect.currentIndexChanged.connect(self.change_classifier)

    def setDataFolder(self):
        self.userList.clear()
        self.usersDir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if self.usersDir:
            # print(arff_filename)
            users = feature_search_start(self.usersDir, False, False, False)
            for user in users:
                self.userList.addItem(user)

    def createDataset(self):
        confirmNewDSet = QMessageBox
        ret = confirmNewDSet.question(self, 'Create new Dataset', 'Warning: Creating a new dataset can take a VERY LONG time! \nAre you sure you want to continue?', confirmNewDSet.Yes | confirmNewDSet.No)

        if ret == confirmNewDSet.Yes:
            arff_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if arff_dir:
                self.currentDataset = create_dataset(feature_search_start(arff_dir, True, True, False))

    def setDataset(self):
        self.currentDataset, _ = create_dataset(QFileDialog.getOpenFileName(self, 'Select Dataset', '.', 'ARFF Datasets (*.arff)'))

    def importCodeFromFile(self):
        self.verifiableCodeEdit.clear()
        codefile = QFileDialog.getOpenFileName(self, 'Select file to import', '.', 'C++ source files (*.cpp)')
        if codefile:
            with open(codefile[0], 'r') as code:
                self.verifiableCodeEdit.setPlainText(code.read())

    def verifyCode(self):
        if self.verifiableCodeEdit.toPlainText():
            with open('FeatureSearch/temp.cpp', 'w') as out:
                out.write(self.verifiableCodeEdit.toPlainText())
            features = get_features_for_file('FeatureSearch/temp.cpp', '')
            if self.userList.selectedItems():
                class_res = compare_to_user_files(features, self.trained_model)
                verificationResult = QMessageBox()
                verificationResult.setIcon(QMessageBox.Information)
                verificationResult.setWindowTitle('Result')
                verificationResult.setText(f'Confidence that the code belongs to selected user: {class_res*100}%.')
                verificationResult.setStandardButtons(QMessageBox.Ok)
                verificationResult.exec_()
            else:
                noUserSelectedWarning = QMessageBox()
                noUserSelectedWarning.setIcon(QMessageBox.Warning)
                noUserSelectedWarning.setWindowTitle('No Target User Selected')
                noUserSelectedWarning.setText('You must select a user to compare the file to.')
                noUserSelectedWarning.setStandardButtons(QMessageBox.Ok)
                noUserSelectedWarning.exec_()
            os.remove('FeatureSearch/temp.cpp')
        else:
            emptyVCodeWarning = QMessageBox()
            emptyVCodeWarning.setIcon(QMessageBox.Critical)
            emptyVCodeWarning.setWindowTitle('Error while verifying')
            emptyVCodeWarning.setText('You must write or import code in the text box in order to verify it.')
            emptyVCodeWarning.setStandardButtons(QMessageBox.Ok)
            emptyVCodeWarning.exec_()

    def showUserFiles(self, item):
        self.codeFileList.clear()
        self.trained_model = train_for_user(self.comboBoxClassifierSelect.currentIndex(), self.userList.currentRow(), self.currentDataset)
        self.selectedUser = f'{self.usersDir}/{str(item.text())}'
        user_files = next(os.walk(self.selectedUser))[2]
        # print(user_files)
        for code in user_files:
            if code[0] != '.':
                self.codeFileList.addItem(code)

    def showCode(self, item):
        self.textBrowserCodeViewer.clear()
        with open(f'{self.selectedUser}/{str(item.text())}', 'r') as src:
            self.textBrowserCodeViewer.setText(src.read())

    def change_classifier(self, index):
        if self.userList.selectedIndexes():
            self.trained_model = train_for_user(index, self.userList.selectedIndexes()[0].row(), self.currentDataset)


def catch_exceptions(t, val, tb):
    QtWidgets.QMessageBox.critical(None,
                                   "An exception was raised",
                                   "Exception type: {}".format(t))
    old_hook(t, val, tb)


if __name__ == "__main__":
    old_hook = sys.excepthook
    sys.excepthook = catch_exceptions
    app = QtWidgets.QApplication(sys.argv)
    FeatureSearchDemo = QtWidgets.QMainWindow()
    ui = Ui_FeatureSearchDemo()
    ui.setupUi(FeatureSearchDemo)
    FeatureSearchDemo.show()
    sys.exit(app.exec_())

