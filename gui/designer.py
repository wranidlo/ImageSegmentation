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
        MainWindow.resize(795, 538)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_10.setObjectName("gridLayout_10")
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
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.tab3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_3.addWidget(self.label_7)
        self.check_box_all = QtWidgets.QCheckBox(self.tab3)
        self.check_box_all.setChecked(True)
        self.check_box_all.setObjectName("check_box_all")
        self.verticalLayout_3.addWidget(self.check_box_all)
        self.check_box_a = QtWidgets.QCheckBox(self.tab3)
        self.check_box_a.setObjectName("check_box_a")
        self.verticalLayout_3.addWidget(self.check_box_a)
        self.check_box_b = QtWidgets.QCheckBox(self.tab3)
        self.check_box_b.setObjectName("check_box_b")
        self.verticalLayout_3.addWidget(self.check_box_b)
        self.gridLayout_3.addLayout(self.verticalLayout_3, 0, 1, 1, 1)
        self.button_clear_segment = QtWidgets.QPushButton(self.tab3)
        self.button_clear_segment.setObjectName("button_clear_segment")
        self.gridLayout_3.addWidget(self.button_clear_segment, 5, 1, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_9 = QtWidgets.QLabel(self.tab3)
        self.label_9.setObjectName("label_9")
        self.gridLayout_4.addWidget(self.label_9, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.tab3)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 0, 0, 1, 1)
        self.list_edges = QtWidgets.QListWidget(self.tab3)
        self.list_edges.setObjectName("list_edges")
        self.gridLayout_4.addWidget(self.list_edges, 1, 0, 1, 1)
        self.list_regions = QtWidgets.QListWidget(self.tab3)
        self.list_regions.setObjectName("list_regions")
        self.gridLayout_4.addWidget(self.list_regions, 1, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_4, 3, 2, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_6 = QtWidgets.QLabel(self.tab3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.check_box_edge_seg = QtWidgets.QCheckBox(self.tab3)
        self.check_box_edge_seg.setObjectName("check_box_edge_seg")
        self.verticalLayout_2.addWidget(self.check_box_edge_seg)
        self.check_box_water = QtWidgets.QCheckBox(self.tab3)
        self.check_box_water.setObjectName("check_box_water")
        self.verticalLayout_2.addWidget(self.check_box_water)
        self.check_box_region = QtWidgets.QCheckBox(self.tab3)
        self.check_box_region.setObjectName("check_box_region")
        self.verticalLayout_2.addWidget(self.check_box_region)
        self.gridLayout_3.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem4, 3, 0, 1, 1)
        self.button_segment = QtWidgets.QPushButton(self.tab3)
        self.button_segment.setObjectName("button_segment")
        self.gridLayout_3.addWidget(self.button_segment, 5, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_10 = QtWidgets.QLabel(self.tab3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout.addWidget(self.label_10)
        self.text_save_location = QtWidgets.QLineEdit(self.tab3)
        self.text_save_location.setObjectName("text_save_location")
        self.horizontalLayout.addWidget(self.text_save_location)
        self.button_save = QtWidgets.QPushButton(self.tab3)
        self.button_save.setObjectName("button_save")
        self.horizontalLayout.addWidget(self.button_save)
        self.gridLayout_3.addLayout(self.horizontalLayout, 5, 2, 1, 1)
        self.progress_bar_seg = QtWidgets.QProgressBar(self.tab3)
        self.progress_bar_seg.setProperty("value", 0)
        self.progress_bar_seg.setObjectName("progress_bar_seg")
        self.gridLayout_3.addWidget(self.progress_bar_seg, 0, 2, 1, 1)
        self.tabWidget.addTab(self.tab3, "")
        self.tab4 = QtWidgets.QWidget()
        self.tab4.setObjectName("tab4")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab4)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 167, 122))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_ev_algorithms = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_ev_algorithms.setFont(font)
        self.label_ev_algorithms.setObjectName("label_ev_algorithms")
        self.verticalLayout_6.addWidget(self.label_ev_algorithms)
        self.check_ev_alg_jaccarda = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.check_ev_alg_jaccarda.setObjectName("check_ev_alg_jaccarda")
        self.verticalLayout_6.addWidget(self.check_ev_alg_jaccarda)
        self.check_ev_alg_f1_score = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.check_ev_alg_f1_score.setObjectName("check_ev_alg_f1_score")
        self.verticalLayout_6.addWidget(self.check_ev_alg_f1_score)
        self.check_ev_alg_evs = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.check_ev_alg_evs.setObjectName("check_ev_alg_evs")
        self.verticalLayout_6.addWidget(self.check_ev_alg_evs)
        self.check_ev_alg_mse = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.check_ev_alg_mse.setObjectName("check_ev_alg_mse")
        self.verticalLayout_6.addWidget(self.check_ev_alg_mse)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab4)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(320, 50, 441, 401))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_ev_results = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_ev_results.setFont(font)
        self.label_ev_results.setObjectName("label_ev_results")
        self.verticalLayout_7.addWidget(self.label_ev_results)
        self.list_ev_results = QtWidgets.QListWidget(self.verticalLayoutWidget_2)
        self.list_ev_results.setObjectName("list_ev_results")
        self.verticalLayout_7.addWidget(self.list_ev_results)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.tab4)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(10, 150, 291, 96))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_ev_true_folder = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_ev_true_folder.setFont(font)
        self.label_ev_true_folder.setObjectName("label_ev_true_folder")
        self.verticalLayout_8.addWidget(self.label_ev_true_folder)
        self.line_ev_true_folder = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.line_ev_true_folder.setObjectName("line_ev_true_folder")
        self.verticalLayout_8.addWidget(self.line_ev_true_folder)
        self.label_ev_pred_folder = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_ev_pred_folder.setFont(font)
        self.label_ev_pred_folder.setObjectName("label_ev_pred_folder")
        self.verticalLayout_8.addWidget(self.label_ev_pred_folder)
        self.line_ev_pred_folder = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.line_ev_pred_folder.setObjectName("line_ev_pred_folder")
        self.verticalLayout_8.addWidget(self.line_ev_pred_folder)
        self.progressBar_evaluation = QtWidgets.QProgressBar(self.tab4)
        self.progressBar_evaluation.setGeometry(QtCore.QRect(320, 10, 451, 23))
        self.progressBar_evaluation.setProperty("value", 0)
        self.progressBar_evaluation.setObjectName("progressBar_evaluation")
        self.pushButton_evaluation = QtWidgets.QPushButton(self.tab4)
        self.pushButton_evaluation.setGeometry(QtCore.QRect(210, 10, 93, 28))
        self.pushButton_evaluation.setObjectName("pushButton_evaluation")
        self.tabWidget.addTab(self.tab4, "")
        self.gridLayout_10.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(3)
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
        self.label_7.setText(_translate("MainWindow", "Wybierz zakres"))
        self.check_box_all.setText(_translate("MainWindow", "Wszystkie"))
        self.check_box_a.setText(_translate("MainWindow", "Grupa pierwsza"))
        self.check_box_b.setText(_translate("MainWindow", "Grupa druga"))
        self.button_clear_segment.setText(_translate("MainWindow", "Wyczyść"))
        self.label_9.setText(_translate("MainWindow", "Pola"))
        self.label_8.setText(_translate("MainWindow", "Krawędzie"))
        self.label_6.setText(_translate("MainWindow", "Wybierz typ"))
        self.check_box_edge_seg.setText(_translate("MainWindow", "Krawędziowa"))
        self.check_box_water.setText(_translate("MainWindow", "Wododziałowa"))
        self.check_box_region.setText(_translate("MainWindow", "Regionowa"))
        self.button_segment.setText(_translate("MainWindow", "Segmentuj"))
        self.label_10.setText(_translate("MainWindow", "Podaj lokalizację zapisu"))
        self.button_save.setText(_translate("MainWindow", "Zapisz wyniki"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab3), _translate("MainWindow", "Segmentacja"))
        self.label_ev_algorithms.setText(_translate("MainWindow", "Wybierz algorytm"))
        self.check_ev_alg_jaccarda.setText(_translate("MainWindow", "Jaccarda Index"))
        self.check_ev_alg_f1_score.setText(_translate("MainWindow", "F1 Score"))
        self.check_ev_alg_evs.setText(_translate("MainWindow", "Explained Varince Score"))
        self.check_ev_alg_mse.setText(_translate("MainWindow", "Mean Squared Error"))
        self.label_ev_results.setText(_translate("MainWindow", "Wyniki"))
        self.label_ev_true_folder.setText(_translate("MainWindow", "Folder z prawdziwą segmentacją"))
        self.line_ev_true_folder.setText(_translate("MainWindow", "D:/GitHub/ImageSegmentation/ground truth/gray/human"))
        self.label_ev_pred_folder.setText(_translate("MainWindow", "Folder z otrzymaną segmentacją"))
        self.line_ev_pred_folder.setText(_translate("MainWindow", "D:/GitHub/ImageSegmentation/results"))
        self.pushButton_evaluation.setText(_translate("MainWindow", "Ewaluacja"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab4), _translate("MainWindow", "Ewaluacja"))
