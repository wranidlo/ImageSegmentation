from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtGui
from gui.designer import Ui_MainWindow
import os.path
import cv2
import glob
import os, os.path


class MainClass(Ui_MainWindow, QMainWindow):
    def __init__(self):
        self.completed = 0
        self.images_list = []
        super(MainClass, self).__init__()
        self.setupUi(self)

        self.model_list_of_images = QtGui.QStandardItemModel()
        self.list_of_images.setModel(self.model_list_of_images)
        self.add_one_image_button.clicked.connect(self.add_one_image)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.delete_button.clicked.connect(self.delete_image_from_list)

    def add_one_image(self):
        img_path = self.one_image_text.text()
        if os.path.isfile(img_path):
            self.wrong_data_error_label.setText("")
            img = cv2.imread(img_path)
            self.images_list.append(img)
            img = cv2.resize(img, (50, 50))
            image = QtGui.QImage(img.data, img.shape[1], img.shape[0],
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
                self.images_list.append(img)
                img = cv2.resize(img, (50, 50))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0],
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


if __name__ == '__main__':
    app = QApplication([])
    vp = MainClass()
    vp.show()
    app.exec_()
