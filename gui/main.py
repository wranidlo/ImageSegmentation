from PyQt5.QtWidgets import QMainWindow, QApplication, QButtonGroup, QListWidgetItem
from PyQt5 import QtGui, QtCore
from gui.designer import Ui_MainWindow
import os.path
import cv2
from group_images import testBlurrLaplacianVariance
import os, os.path


class MainClass(Ui_MainWindow, QMainWindow):
    def __init__(self):
        self.completed = 0
        self.images_divided = False
        self.images_list = []   # list of loaded images names
        self.images_group_a = []
        self.images_group_b = []
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
            lenght = len(os.listdir(img_folder))
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
                self.completed += (int)(100/lenght)
                self.progress_bar_list.setValue(self.completed)
            self.progress_bar_list.setValue(100)
        else:
            self.wrong_data_error_label.setStyleSheet("color:red")
            self.wrong_data_error_label.setText("Brak folderu o podanej nazwie")

    def delete_image_from_list(self):
        self.model_list_of_images.clear()
        self.images_list.clear()

    def divide_images(self):
        self.images_group_a.clear()
        self.images_group_b.clear()
        self.list_widget_a.clear()
        self.list_widget_b.clear()
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

                    self.images_group_b.append(e)
                    self.list_widget_b.addItem(item)


if __name__ == '__main__':
    app = QApplication([])
    vp = MainClass()
    vp.show()
    app.exec_()
