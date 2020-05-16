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
import mainTemp  # TEMP
import evaluation
import watershedSegmentation

from matplotlib import pyplot as plt

sys.path.append('../')


class MainClass(Ui_MainWindow, QMainWindow):
    def __init__(self):
        self.completed = 0  # status bar
        self.images_divided = False
        self.images_list = []  # list of loaded images names
        self.images_group_a = []  # list of images (group a) exists only if images where divided
        self.images_group_b = []  # list of images (group b) exists only if images where divided
        self.list_to_segment = []   # list of images names to segment
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
        self.button_save.clicked.connect(self.save_seg)

        self.seg_type = ""

        #  4 tab
        self.pushButton_evaluation.clicked.connect(self.evaluation)

    def add_one_image(self):
        img_path = self.one_image_text.text()
        if os.path.isfile(img_path):
            self.wrong_data_error_label.setText("")
            img = cv2.imread(img_path)
            self.images_list.append(img_path)
            img = cv2.resize(img, (50, 50))
            image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
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
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QtGui.QStandardItem(f)
                item.setIcon(icon)
                self.model_list_of_images.appendRow(item)
                self.completed += (int)(100 / length)
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
        self.list_to_segment = []
        self.list_regions.clear()
        self.list_edges.clear()
        self.edges_list.clear()
        self.regions_list.clear()
        self.completed = 0
        self.progress_bar_seg.setValue(0)
        if self.check_box_all.isChecked():
            self.list_to_segment = self.images_list
        if self.check_box_a.isChecked():
            self.list_to_segment = self.images_group_a
        if self.check_box_b.isChecked():
            self.list_to_segment = self.images_group_b
        length = len(self.list_to_segment)
        if self.check_box_edge_seg.isChecked():
            self.seg_type = "Edge"
            for e in self.list_to_segment:
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
        if self.check_box_water.isChecked():
            self.seg_type = "Watershed"
            for e in self.list_to_segment:
                segmentation = watershedSegmentation.watershedSegmentation(e)

                edges, regions = segmentation.getResults()
                plt.imsave('images/temp.png', regions)
                regions = cv2.imread('images/temp.png')
                image = QtGui.QImage(edges.data, edges.shape[1], edges.shape[0], edges.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(e)
                item.setIcon(icon)
                self.list_edges.addItem(item)
                self.edges_list.append(edges)
                image = QtGui.QImage(regions.data, regions.shape[1], regions.shape[0], 3 * regions.shape[1],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
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
        self.list_to_segment.clear()

    def save_seg(self):
        img_folder = self.text_save_location.text()
        if os.path.exists(img_folder):
            for i in range(0, len(self.list_to_segment)):
                words = self.list_to_segment[i].split('\\')
                fileAndExtension = words[len(words) - 1]
                fileAndExtensionList = fileAndExtension.split('.')
                fileName = fileAndExtensionList[0]
                extensionName = fileAndExtensionList[1]
                binaryPath = img_folder + '\\' + fileName + "-" +self.seg_type + "Binary." + extensionName
                markerPath = img_folder + '\\' + fileName + "-" +self.seg_type + "Marker.png"
                cv2.imwrite(binaryPath, self.edges_list[i])
                plt.imsave(markerPath, self.regions_list[i])

    def evaluation(self):
        self.list_ev_results.clear()
        self.completed = 0
        self.progressBar_evaluation.setValue(0)
        algorithms = [0, 0, 0, 0]
        if self.check_ev_alg_jaccarda.isChecked():
            algorithms[0] = 1
        if self.check_ev_alg_f1_score.isChecked():
            algorithms[1] = 1
        if self.check_ev_alg_evs.isChecked():
            algorithms[2] = 1
        if self.check_ev_alg_mse.isChecked():
            algorithms[3] = 1
        ground_true_folder = self.line_ev_true_folder.text()
        ground_true_folder += "/*.bmp"
        true_segmentation_paths = mainTemp.loadImagesPathsInFolder(ground_true_folder)
        print("folder: ", ground_true_folder)
        print("paths: ", true_segmentation_paths)
        predicted_folder = self.line_ev_pred_folder.text()
        predicted_folder += "/*.jpg"
        print("folder: ", predicted_folder)
        predicted_segmentation_paths = mainTemp.loadImagesPathsInFolder(predicted_folder)
        print("before ", predicted_segmentation_paths)
        predicted_segmentation_paths = self.filterPredictedPathsForEvaluation(predicted_segmentation_paths, 'Binary')
        print("after ", predicted_segmentation_paths)
        evaluation_paths = self.connectPaths(predicted_segmentation_paths, true_segmentation_paths)
        evaluation_results = []
        length = len(evaluation_paths)
        for path in evaluation_paths:
            # TODO fix displaying many elements in list row or maybe change on QTable
            # Evaluation
            ev = evaluation.Evaluation(evaluation_paths, algorithms)
            evaluation_results.append(ev.getResults())
            # Displaying result
            pred_img = cv2.imread(evaluation_results[0]["predicted path"])
            true_img = cv2.imread(evaluation_results[0]["true path"])
            pred_img = cv2.resize(pred_img, (50, 50))
            true_img = cv2.resize(true_img, (50, 50))
            pred_img = QtGui.QImage(pred_img.data, pred_img.shape[1], pred_img.shape[0], 3 * pred_img.shape[1],
                                    QtGui.QImage.Format_RGB888)
            true_img = QtGui.QImage(true_img.data, true_img.shape[1], true_img.shape[0], 3 * true_img.shape[1],
                                    QtGui.QImage.Format_RGB888)
            pred_icon = QtGui.QIcon()
            true_icon = QtGui.QIcon()
            pred_icon.addPixmap(QtGui.QPixmap.fromImage(pred_img), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            true_icon.addPixmap(QtGui.QPixmap.fromImage(true_img), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item = QListWidgetItem(pred_icon, true_icon, evaluation_results[0]["predicted path"], evaluation_results[0]
            ["jaccarda index weighted"], evaluation_results[0]["f1 score weighted"], evaluation_results[0]
                                   ["explained variance score"], evaluation_results[0]["mean squared error"])
            # item.setIcon(icon)
            self.list_ev_results.addItem(item)
            # Update progress bar
            self.completed += (int)(100 / length)
            self.progress_bar_seg.setValue(self.completed)
        self.progress_bar_seg.setValue(100)
        # TODO diplaying results

    def filterPredictedPathsForEvaluation(self, paths, keyString):
        filtered_paths = [path for path in paths if keyString in path]
        return filtered_paths

    def connectPaths(self, predicted, true):
        connectedPaths = []
        for pred_path in predicted:
            name = self.getImageNameFromFilePath(pred_path)
            true_path = self.findGroundTruthForImage(name, true)
            # TODO exception not found
            if true_path == "":
                None
            # TODO
            connectedPaths.append([true_path, pred_path])
        return connectedPaths

    def getImageNameFromFilePath(self, filePath):
        words = self.originalPath.split('/')
        fileAndExtension = words[len(words) - 1]
        fileAndExtensionList = fileAndExtension.split('.')
        fileName = fileAndExtensionList[0]
        extensionName = fileAndExtensionList[1]
        return fileName

    def findGroundTruthForImage(self, imageName, groundTruthPaths):
        for path in groundTruthPaths:
            fileName = self.getImageNameFromFilePath(path)
            foundName = (fileName.split('-'))[0]
            if foundName == imageName:
                return path
        return ""


def main():
    app = QApplication([])
    vp = MainClass()
    vp.show()
    app.exec_()


if __name__ == '__main__':
    main()
