from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import Covid19.Covid19Classification as Covid19Classification
import os
import cv2
import random
import numpy as np


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(637, 432)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setIconSize(QtCore.QSize(75, 75))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(160, 10, 261, 51))
        self.Title.setObjectName("Title")
        self.TrainButton = QtWidgets.QPushButton(self.centralwidget)
        self.TrainButton.setGeometry(QtCore.QRect(10, 80, 71, 31))
        self.TrainButton.setObjectName("TrainButton")
        self.TestButton = QtWidgets.QPushButton(self.centralwidget)
        self.TestButton.setGeometry(QtCore.QRect(50, 130, 75, 23))
        self.TestButton.setObjectName("TestButton")
        self.RocButton = QtWidgets.QPushButton(self.centralwidget)
        self.RocButton.setGeometry(QtCore.QRect(10, 170, 81, 31))
        self.RocButton.setObjectName("RocButton")
        self.CMetricButton = QtWidgets.QPushButton(self.centralwidget)
        self.CMetricButton.setGeometry(QtCore.QRect(100, 170, 81, 31))
        self.CMetricButton.setObjectName("CMetricButton")
        self.IndexButton = QtWidgets.QPushButton(self.centralwidget)
        self.IndexButton.setGeometry(QtCore.QRect(50, 220, 81, 31))
        self.IndexButton.setObjectName("IndexButton")
        self.LoadParams = QtWidgets.QPushButton(self.centralwidget)
        self.LoadParams.setGeometry(QtCore.QRect(90, 80, 91, 31))
        self.LoadParams.setObjectName("LoadParams")
        self.XrayImage = QtWidgets.QLabel(self.centralwidget)
        self.XrayImage.setGeometry(QtCore.QRect(380, 50, 241, 311))
        self.XrayImage.setAutoFillBackground(True)
        self.XrayImage.setText("")
        self.XrayImage.setObjectName("XrayImage")
        self.GetXrayImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.GetXrayImageButton.setGeometry(QtCore.QRect(210, 290, 81, 41))
        self.GetXrayImageButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.GetXrayImageButton.setObjectName("GetXrayImageButton")
        self.TrainImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.TrainImageButton.setGeometry(QtCore.QRect(210, 350, 75, 23))
        self.TrainImageButton.setObjectName("TrainImageButton")
        self.TrueLabel = QtWidgets.QLabel(self.centralwidget)
        self.TrueLabel.setGeometry(QtCore.QRect(210, 80, 121, 31))
        self.TrueLabel.setObjectName("TrueLabel")
        self.PredLabel = QtWidgets.QLabel(self.centralwidget)
        self.PredLabel.setGeometry(QtCore.QRect(210, 140, 121, 31))
        self.PredLabel.setObjectName("PredLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # 自写代码
        self.flag = 0  # 判断是否有训练
        self.flag2 = 0  # 是否有图片
        self.TrainButton.clicked.connect(self.TrainButton_Click)
        self.LoadParams.clicked.connect(self.LoadParams_Click)
        self.TestButton.clicked.connect(self.TestButton_Click)
        self.RocButton.clicked.connect(self.RocButton_Click)
        self.CMetricButton.clicked.connect(self.CMetricButton_Click)
        self.IndexButton.clicked.connect(self.IndexButton_Click)
        self.GetXrayImageButton.clicked.connect(self.GetXrayImageButton_Click)
        self.TrainImageButton.clicked.connect(self.TrainImageButton_Click)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def GetXrayImageButton_Click(self):
        input_dir = "dataset/"
        positive_file_dirs = [input_dir + "covid/" + filename for filename in os.listdir(input_dir + "covid/")
                              if ("jpeg" in filename or "jpg" in filename or "png" in filename)]
        negative_file_dirs = [input_dir + "normal/" + filename for filename in os.listdir(input_dir + "normal/")
                              if ("jpeg" in filename or "jpg" in filename or "png" in filename)]
        all_file_dirs = positive_file_dirs + negative_file_dirs
        file_dirs = all_file_dirs[random.sample(range(0, len(all_file_dirs)), 1)[0]]
        if file_dirs.split("/")[1] == "normal":
            y_true = 0
        else:
            y_true = 1
        print(file_dirs, y_true)
        self.image1 = cv2.resize(cv2.imread(file_dirs), (128, 128), interpolation=cv2.INTER_CUBIC)
        self.flag2 = 1
        self.TrueLabel.setText('True: Covid19' if y_true == 1 else 'True: Normal')
        pix = QPixmap(file_dirs)
        pix = pix.scaled(251, 271)
        self.XrayImage.setPixmap(pix)

    def TrainImageButton_Click(self):
        if self.flag == 0:
            print("请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!")
        elif self.flag2 == 0:
            print("请先获取X光图片\n请先获取X光图片\n请先获取X光图片\n请先获取X光图片\n请先获取X光图片\n")
        else:
            y_score = Covid19Classification.predict(self.model, self.image1)
            y_pred = np.rint(y_score)
            self.PredLabel.setText("Pred: Covid19" if int(y_pred[0]) == 1 else "Pred: Normal")

    def TrainButton_Click(self):
        self.model, self.X_test, self.y_test = Covid19Classification.train(load_params=False)
        self.flag = 1

    def LoadParams_Click(self):
        self.model, self.X_test, self.y_test = Covid19Classification.train(load_params=True)
        self.flag = 1
        print("模型已加载!!\n模型已加载!!\n模型已加载!!\n模型已加载!!\n模型已加载!!\n")

    def TestButton_Click(self):
        if self.flag == 0:
            print("请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!\n请先训练好模型!")
        else:
            Covid19Classification.test(self.model, self.X_test, self.y_test)

    def RocButton_Click(self):
        Covid19Classification.show_anything(self.model, self.X_test, self.y_test, flag=1)

    def CMetricButton_Click(self):
        Covid19Classification.show_anything(self.model, self.X_test, self.y_test, flag=2)

    def IndexButton_Click(self):
        Covid19Classification.show_anything(self.model, self.X_test, self.y_test, flag=3)

    # end

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Covid19_Classification v1.1 --by Chino"))
        self.Title.setText(_translate("MainWindow", "基于深度学习的X光片分类器(是否患有Covid19)"))
        self.TrainButton.setText(_translate("MainWindow", "训练模型"))
        self.TestButton.setText(_translate("MainWindow", "测试模型"))
        self.RocButton.setText(_translate("MainWindow", "显示ROC曲线"))
        self.CMetricButton.setText(_translate("MainWindow", "显示混淆矩阵"))
        self.IndexButton.setText(_translate("MainWindow", "输出各项指标"))
        self.LoadParams.setText(_translate("MainWindow", "加载已训练参数"))
        self.GetXrayImageButton.setText(_translate("MainWindow", "获取X光图片"))
        self.TrainImageButton.setText(_translate("MainWindow", "训练"))
        self.TrueLabel.setText(_translate("MainWindow", "待定"))
        self.PredLabel.setText(_translate("MainWindow", "待定"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
sys.exit(app.exec_())
