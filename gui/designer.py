# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designer.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(887, 591)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab1 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab1.sizePolicy().hasHeightForWidth())
        self.tab1.setSizePolicy(sizePolicy)
        self.tab1.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.tab1.setObjectName("tab1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.delete_button = QtWidgets.QPushButton(self.tab1)
        self.delete_button.setObjectName("delete_button")
        self.gridLayout.addWidget(self.delete_button, 6, 2, 1, 1)
        self.folder_text = QtWidgets.QLineEdit(self.tab1)
        self.folder_text.setText("")
        self.folder_text.setObjectName("folder_text")
        self.gridLayout.addWidget(self.folder_text, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 4, 0, 1, 1)
        self.progress_bar_list = QtWidgets.QProgressBar(self.tab1)
        self.progress_bar_list.setProperty("value", 0)
        self.progress_bar_list.setObjectName("progress_bar_list")
        self.gridLayout.addWidget(self.progress_bar_list, 3, 2, 1, 1)
        self.add_one_image_button = QtWidgets.QPushButton(self.tab1)
        self.add_one_image_button.setObjectName("add_one_image_button")
        self.gridLayout.addWidget(self.add_one_image_button, 2, 0, 1, 1)
        self.one_image_text = QtWidgets.QLineEdit(self.tab1)
        self.one_image_text.setText("")
        self.one_image_text.setObjectName("one_image_text")
        self.gridLayout.addWidget(self.one_image_text, 1, 0, 1, 1)
        self.add_folder_button = QtWidgets.QPushButton(self.tab1)
        self.add_folder_button.setObjectName("add_folder_button")
        self.gridLayout.addWidget(self.add_folder_button, 2, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 5, 2, 1, 1)
        self.list_of_images = QtWidgets.QListView(self.tab1)
        self.list_of_images.setViewMode(QtWidgets.QListView.ListMode)
        self.list_of_images.setObjectName("list_of_images")
        self.gridLayout.addWidget(self.list_of_images, 4, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.wrong_data_error_label = QtWidgets.QLabel(self.tab1)
        self.wrong_data_error_label.setText("")
        self.wrong_data_error_label.setAlignment(QtCore.Qt.AlignCenter)
        self.wrong_data_error_label.setObjectName("wrong_data_error_label")
        self.gridLayout.addWidget(self.wrong_data_error_label, 2, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab1, "")
        self.tab2 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab2.sizePolicy().hasHeightForWidth())
        self.tab2.setSizePolicy(sizePolicy)
        self.tab2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tab2.setObjectName("tab2")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_17 = QtWidgets.QLabel(self.tab2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.verticalLayout.addWidget(self.label_17)
        self.check_box_edge = QtWidgets.QCheckBox(self.tab2)
        self.check_box_edge.setObjectName("check_box_edge")
        self.verticalLayout.addWidget(self.check_box_edge)
        self.check_box_blurr = QtWidgets.QCheckBox(self.tab2)
        self.check_box_blurr.setObjectName("check_box_blurr")
        self.verticalLayout.addWidget(self.check_box_blurr)
        self.button_divide = QtWidgets.QPushButton(self.tab2)
        self.button_divide.setObjectName("button_divide")
        self.verticalLayout.addWidget(self.button_divide)
        self.progress_bar_groups = QtWidgets.QProgressBar(self.tab2)
        self.progress_bar_groups.setProperty("value", 0)
        self.progress_bar_groups.setObjectName("progress_bar_groups")
        self.verticalLayout.addWidget(self.progress_bar_groups)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.label_5 = QtWidgets.QLabel(self.tab2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.button_clear_divide = QtWidgets.QPushButton(self.tab2)
        self.button_clear_divide.setObjectName("button_clear_divide")
        self.verticalLayout.addWidget(self.button_clear_divide)
        self.gridLayout_5.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridLayout_11 = QtWidgets.QGridLayout()
        self.gridLayout_11.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_group_a = QtWidgets.QLabel(self.tab2)
        self.label_group_a.setObjectName("label_group_a")
        self.gridLayout_11.addWidget(self.label_group_a, 0, 0, 1, 1)
        self.label_group_b = QtWidgets.QLabel(self.tab2)
        self.label_group_b.setObjectName("label_group_b")
        self.gridLayout_11.addWidget(self.label_group_b, 0, 1, 1, 1)
        self.list_widget_a = QtWidgets.QListWidget(self.tab2)
        self.list_widget_a.setObjectName("list_widget_a")
        self.gridLayout_11.addWidget(self.list_widget_a, 1, 0, 1, 1)
        self.list_widget_b = QtWidgets.QListWidget(self.tab2)
        self.list_widget_b.setObjectName("list_widget_b")
        self.gridLayout_11.addWidget(self.list_widget_b, 1, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_11, 0, 2, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem3, 0, 1, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab2, "")
        self.tab3 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab3.sizePolicy().hasHeightForWidth())
        self.tab3.setSizePolicy(sizePolicy)
        self.tab3.setObjectName("tab3")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab3)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 691, 491))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabWidget.addTab(self.tab3, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        self.gridLayout_7.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image segmentation"))
        self.delete_button.setText(_translate("MainWindow", "Usuń"))
        self.folder_text.setPlaceholderText(_translate("MainWindow", "example.jpg"))
        self.label_2.setText(_translate("MainWindow", "Lista obrazów"))
        self.add_one_image_button.setText(_translate("MainWindow", "Dodaj"))
        self.one_image_text.setPlaceholderText(_translate("MainWindow", "example.jpg"))
        self.add_folder_button.setText(_translate("MainWindow", "Dodaj"))
        self.label_4.setText(_translate("MainWindow", "Wyczyść listę obrazów"))
        self.label.setText(_translate("MainWindow", "Podaj lokalizację obrazu"))
        self.label_3.setText(_translate("MainWindow", "Podaj  folder z obrazami"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), _translate("MainWindow", "Dodaj obrazy"))
        self.label_17.setText(_translate("MainWindow", "Wybierz metodę podziału"))
        self.check_box_edge.setText(_translate("MainWindow", "Podział według krawedzi"))
        self.check_box_blurr.setText(_translate("MainWindow", "Podział według rozmycia"))
        self.button_divide.setText(_translate("MainWindow", "Podziel"))
        self.label_5.setText(_translate("MainWindow", "Wyczyść podział"))
        self.button_clear_divide.setText(_translate("MainWindow", "Wyczyść"))
        self.label_group_a.setText(_translate("MainWindow", "Grupa pierwsza"))
        self.label_group_b.setText(_translate("MainWindow", "Grupa druga"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2), _translate("MainWindow", "Grupowanie obrazów"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab3), _translate("MainWindow", "Segmentacja"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Strona"))
