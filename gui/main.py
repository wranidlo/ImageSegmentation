import os
import os.path
import os.path
import sys

import cv2
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QButtonGroup, QListWidgetItem

from group_images import testBlurrLaplacianVariance, testEdgesKMeans
from gui.designer import Ui_MainWindow
import edgeBasedSegmentation
sys.path.append('../')


class MainClass(Ui_MainWindow, QMainWindow):
    def __init__(self):
        self.completed = 0
        self.images_divided = False
        self.images_list = []   # list of loaded images names
        self.images_group_a = []    # list of images (group a) exists only if images where divided
        self.images_group_b = []    # list of images (group b) exists only if images where divided
        super(MainClass, self).__init__()
        self.setupUi(self)

        # 1 tab
        self.model_list_of_images = QtGui.QStandardItemModel()
        self.list_of_images.setModel(self.model_list_of_images)
        self.list_of_images.setIconSize(QtCore.QSize(50, 50))

        self.add_one_image_button.clicked.connect(self.add_one_image)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.delete_button.clicked.connect(self.delete_image_from_list)

        # 2 tab
        self.bg = QButtonGroup()
        self.bg.addButton(self.check_box_edge, 1)
        self.bg.addButton(self.check_box_blurr, 2)
        self.list_widget_b.setIconSize(QtCore.QSize(50, 50))
        self.list_widget_a.setIconSize(QtCore.QSize(50, 50))

        self.button_divide.clicked.connect(self.divide_images)
        self.button_clear_divide.clicked.connect(self.clear_divide)

        # 3 tab
        self.bg2 = QButtonGroup()
        self.bg2.addButton(self.check_box_edge_seg, 1)
        self.bg2.addButton(self.check_box_region, 2)
        self.bg2.addButton(self.check_box_water, 3)
        self.bg3 = QButtonGroup()
        self.bg3.addButton(self.check_box_all, 1)
        self.bg3.addButton(self.check_box_a, 2)
        self.bg3.addButton(self.check_box_b, 3)
        self.list_edges.setIconSize(QtCore.QSize(100, 100))
        self.list_regions.setIconSize(QtCore.QSize(100, 100))

        self.edges_list = []
        self.regions_list = []

        self.button_segment.clicked.connect(self.segmentation)
        self.button_clear_segment.clicked.connect(self.clear_seg)

    def add_one_image(self):
        img_path = self.one_image_text.text()
        if os.path.isfile(img_path):
            self.wrong_data_error_label.setText("")
            img = cv2.imread(img_path)
            self.images_list.append(img_path)
            img = cv2.resize(img, (50, 50))
            image = QtGui.QImage(img.data, img.shape[1], img.shape[0],  3*img.shape[1],
                                      QtGui.QImage.Format_RGB888).rgbSwapped()
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item = QtGui.QStandardItem(img_path)
            item.setIcon(icon)
            self.model_list_of_images.appendRow(item)
        else:
            self.wrong_data_error_label.setStyleSheet("color:red")
            self.wrong_data_error_label.setText("Brak pliku o podanej nazwie")

    def add_folder(self):
        self.completed = 0
        self.progress_bar_list.setValue(0)
        img_folder = self.folder_text.text()
        img_folder = img_folder.replace('\\', '/')
        valid_images = [".jpg", ".png"]
        if os.path.exists(img_folder):
            self.wrong_data_error_label.setText("")
            length = len(os.listdir(img_folder))
            for f in os.listdir(img_folder):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                f = os.path.join(img_folder, f)
                img = cv2.imread(f)
                self.images_list.append(f)
                img = cv2.resize(img, (50, 50))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3*img.shape[1],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QtGui.QStandardItem(f)
                item.setIcon(icon)
                self.model_list_of_images.appendRow(item)
                self.completed += (int)(100/length)
                self.progress_bar_list.setValue(self.completed)
            self.progress_bar_list.setValue(100)
        else:
            self.wrong_data_error_label.setStyleSheet("color:red")
            self.wrong_data_error_label.setText("Brak folderu o podanej nazwie")

    def delete_image_from_list(self):
        self.model_list_of_images.clear()
        self.images_list.clear()

    def divide_images(self):
        self.images_divided = True
        self.completed = 0
        self.progress_bar_groups.setValue(0)
        self.images_group_a.clear()
        self.images_group_b.clear()
        self.list_widget_a.clear()
        self.list_widget_b.clear()
        length = len(self.images_list)
        if self.check_box_blurr.isChecked():
            for e in self.images_list:
                if testBlurrLaplacianVariance.test_if_not_blurred(e):
                    self.images_group_a.append(e)

                    img = cv2.imread(e)
                    img = cv2.resize(img, (50, 50))
                    image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3*img.shape[1],
                                         QtGui.QImage.Format_RGB888)
                    icon = QtGui.QIcon()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    item = QListWidgetItem(e)
                    item.setIcon(icon)

                    self.list_widget_a.addItem(item)
                else:
                    self.images_group_b.append(e)

                    img = cv2.imread(e)
                    img = cv2.resize(img, (50, 50))
                    image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3*img.shape[1],
                                         QtGui.QImage.Format_RGB888).rgbSwapped()
                    icon = QtGui.QIcon()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    item = QListWidgetItem(e)
                    item.setIcon(icon)

                    self.list_widget_b.addItem(item)
                self.completed += (int)(100 / length)
                self.progress_bar_groups.setValue(self.completed)
            self.progress_bar_groups.setValue(100)
        if self.check_box_edge.isChecked():
            model = testEdgesKMeans.read_model()
            for e in self.images_list:
                img = cv2.imread(e)
                if testEdgesKMeans.test_group(img, model) == 0:
                    self.images_group_a.append(e)

                    img = cv2.imread(e)
                    img = cv2.resize(img, (50, 50))
                    image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                                         QtGui.QImage.Format_RGB888)
                    icon = QtGui.QIcon()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    item = QListWidgetItem(e)
                    item.setIcon(icon)

                    self.list_widget_a.addItem(item)
                else:
                    self.images_group_b.append(e)

                    img = cv2.imread(e)
                    img = cv2.resize(img, (50, 50))
                    image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                                         QtGui.QImage.Format_RGB888).rgbSwapped()
                    icon = QtGui.QIcon()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    item = QListWidgetItem(e)
                    item.setIcon(icon)

                    self.list_widget_b.addItem(item)
                self.completed += (int)(100 / length)
                self.progress_bar_groups.setValue(self.completed)
            self.progress_bar_groups.setValue(100)

    def clear_divide(self):
        self.images_divided = False
        self.progress_bar_groups.setValue(0)
        self.images_group_a.clear()
        self.images_group_b.clear()
        self.list_widget_a.clear()
        self.list_widget_b.clear()

    def segmentation(self):
        list_to_segment = []
        self.list_regions.clear()
        self.list_edges.clear()
        self.edges_list.clear()
        self.regions_list.clear()
        self.completed = 0
        self.progress_bar_seg.setValue(0)
        if self.check_box_all.isChecked():
            list_to_segment = self.images_list
        if self.check_box_a.isChecked():
            list_to_segment = self.images_group_a
        if self.check_box_b.isChecked():
            list_to_segment = self.images_group_b
        length = len(list_to_segment)
        if self.check_box_edge_seg.isChecked():
            for e in list_to_segment:
                edges, regions = edgeBasedSegmentation.segment_image(e)

                image = QtGui.QImage(edges.data, edges.shape[1], edges.shape[0], edges.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(e)
                item.setIcon(icon)
                self.list_edges.addItem(item)
                self.edges_list.append(edges)

                image = QtGui.QImage(regions.data, regions.shape[1], regions.shape[0], regions.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(e)
                item.setIcon(icon)
                self.list_regions.addItem(item)
                self.regions_list.append(regions)

                self.completed += (int)(100 / length)
                self.progress_bar_seg.setValue(self.completed)
            self.progress_bar_seg.setValue(100)


    def clear_seg(self):
        self.list_regions.clear()
        self.list_edges.clear()
        self.edges_list.clear()
        self.regions_list.clear()
        self.progress_bar_seg.setValue(0)


if __name__ == '__main__':
    app = QApplication([])
    vp = MainClass()
    vp.show()
    app.exec_()
