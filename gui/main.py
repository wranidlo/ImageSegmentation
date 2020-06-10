import os
import os.path
import os.path
import sys
import csv

import cv2
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QButtonGroup, QListWidgetItem, QTableWidgetItem, \
    QHeaderView
from matplotlib import pyplot as plt

import Filters
import Noise
import RegionSegmentation, Region
import edgeBasedSegmentation
import evaluation
import watershedSegmentation
from group_images import testBlurrLaplacianVariance, testEdgesKMeans
from gui.designer import Ui_MainWindow

sys.path.append('../')


class MainClass(Ui_MainWindow, QMainWindow):
    def __init__(self):
        self.completed = 0  # status bar
        self.images_divided = False
        self.images_list = []  # list of loaded images names
        self.images_group_a = []  # list of images (group a) exists only if images where divided
        self.images_group_b = []  # list of images (group b) exists only if images where divided
        self.list_to_segment = []   # list of images names to segment
        self.evaluation_results = [] # list od dicts returned by evaluation
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
        self.filtered_images = []
        self.list_to_filter = []
        self.filter_type = ""
        self.bg_filter = QButtonGroup()
        self.bg_filter.addButton(self.radio_button_bilateral, 1)
        self.bg_filter.addButton(self.radio_button_gaussian, 2)
        self.bg_filter.addButton(self.radio_button_median, 3)
        self.bg_filter.addButton(self.radio_noisegauss, 4)
        self.bg_filter.addButton(self.radio_saltpeper, 5)
        self.bg_filter.addButton(self.radio_speckle, 6)
        self.list_widget_filter.setIconSize(QtCore.QSize(150, 150))
        self.push_button_filter.clicked.connect(self.run_filter)
        self.push_button_save_filter.clicked.connect(self.save_filter)
        self.list_widget_filter.itemDoubleClicked.connect(self.show_clicked_filter)

        # 3 tab
        self.bg = QButtonGroup()
        self.bg.addButton(self.check_box_edge, 1)
        self.bg.addButton(self.check_box_blurr, 2)
        self.list_widget_b.setIconSize(QtCore.QSize(50, 50))
        self.list_widget_a.setIconSize(QtCore.QSize(50, 50))
        self.list_widget_a.itemDoubleClicked.connect(self.show_clicked_a)
        self.list_widget_b.itemDoubleClicked.connect(self.show_clicked_b)

        self.button_divide.clicked.connect(self.divide_images)
        self.button_clear_divide.clicked.connect(self.clear_divide)

        # 4 tab
        self.bg2 = QButtonGroup()
        self.bg2.addButton(self.check_box_edge_seg, 1)
        self.bg2.addButton(self.check_box_region, 2)
        self.bg2.addButton(self.check_box_water, 3)
        self.bg3 = QButtonGroup()
        self.bg3.addButton(self.check_box_all, 1)
        self.bg3.addButton(self.check_box_a, 2)
        self.bg3.addButton(self.check_box_b, 3)
        self.list_widget_edges.setIconSize(QtCore.QSize(100, 100))
        self.list_widget_regions.setIconSize(QtCore.QSize(100, 100))
        self.list_widget_edges.itemDoubleClicked.connect(self.show_clicked_edges)
        self.list_widget_regions.itemDoubleClicked.connect(self.show_clicked_regions)

        self.edges_list = []
        self.regions_list = []

        self.button_segment.clicked.connect(self.segmentation)
        self.button_clear_segment.clicked.connect(self.clear_seg)
        self.button_save.clicked.connect(self.save_seg)

        self.seg_type = ""

        #  5 tab
        self.pushButton_evaluation.clicked.connect(self.evaluation)
        self.pushButton_save_ev.clicked.connect(self.saveEvaluationResults)
        self.table_ev_results.setIconSize(QtCore.QSize(100, 100))
        self.table_ev_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_ev_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.combo_groups.addItem("Edges-Much")
        self.combo_groups.addItem("Edges-Little")
        self.combo_groups.addItem("Blurr")
        self.combo_groups.addItem("Not-Blur")
        self.combo_groups.addItem("Without group")
        self.combo_filter.addItem("Gaussian")
        self.combo_filter.addItem("Bilateral")
        self.combo_filter.addItem("Median")
        self.combo_filter.addItem("Without filter")
        self.combo_noise.addItem("Gaussian")
        self.combo_noise.addItem("Salt-Peper")
        self.combo_noise.addItem("Speckle")
        self.combo_noise.addItem("Without noise")
        self.combo_seg.addItem("EdgesBased")
        self.combo_seg.addItem("RegionBased")
        self.combo_seg.addItem("Watershed")
        self.combo_seg.addItem("Without segmentation")


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
            item = QtGui.QStandardItem(self.getShortFilePath(img_path))
            item.setIcon(icon)
            self.model_list_of_images.appendRow(item)
        else:
            self.wrong_data_error_label.setStyleSheet("color:red")
            self.wrong_data_error_label.setText("Brak pliku o podanej nazwie")

    def add_folder(self):
        self.completed = 0
        self.progress_bar_list.setValue(0)
        img_folder = self.folder_text.text()
        valid_images = [".jpg", ".png", ".bmp"]
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
                item = QtGui.QStandardItem(self.getShortFilePath(f))
                item.setIcon(icon)
                self.model_list_of_images.appendRow(item)
                self.completed += int(100 / length)
                self.progress_bar_list.setValue(self.completed)
            self.progress_bar_list.setValue(100)
        else:
            self.wrong_data_error_label.setStyleSheet("color:red")
            self.wrong_data_error_label.setText("Brak folderu o podanej nazwie")

    def delete_image_from_list(self):
        self.model_list_of_images.clear()
        self.images_list.clear()

    def run_filter(self):
        self.list_to_filter = self.images_list
        self.filtered_images.clear()
        self.list_widget_filter.clear()
        self.progress_bar_filter.setValue(0)
        self.completed = 0
        length = len(self.images_list)
        if self.radio_button_median.isChecked():
            self.filter_type = "Median"
            for e in self.images_list:
                img = cv2.imread(e)
                img = Filters.Median_filter(img, 3)
                self.filtered_images.append(img)
                img = cv2.resize(img, (150, 150))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)

                self.list_widget_filter.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_filter.setValue(self.completed)
            self.progress_bar_filter.setValue(100)
        if self.radio_button_gaussian.isChecked():
            self.filter_type = "Gaussian"
            for e in self.images_list:
                img = cv2.imread(e)
                img = Filters.GaussianBlurImage(img, 3)
                self.filtered_images.append(img)
                img = cv2.resize(img, (150, 150))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)

                self.list_widget_filter.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_filter.setValue(self.completed)
            self.progress_bar_filter.setValue(100)
        if self.radio_button_bilateral.isChecked():
            self.filter_type = "Bilateral"
            for e in self.images_list:
                img = cv2.imread(e)
                img = Filters.Bilateral_filter(img, 30, 30)
                self.filtered_images.append(img)
                img = cv2.resize(img, (150, 150))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)

                self.list_widget_filter.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_filter.setValue(self.completed)
            self.progress_bar_filter.setValue(100)
        if self.radio_speckle.isChecked():
            self.filter_type = "SpeckleNoise"
            for e in self.images_list:
                img = cv2.imread(e)
                img = Noise.noise_Speckle(img)
                self.filtered_images.append(img)
                img = cv2.resize(img, (150, 150))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)

                self.list_widget_filter.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_filter.setValue(self.completed)
            self.progress_bar_filter.setValue(100)
        if self.radio_noisegauss.isChecked():
            self.filter_type = "GaussianNoise"
            for e in self.images_list:
                img = cv2.imread(e)
                img = Noise.noise_Gaussian(img)
                self.filtered_images.append(img)
                img = cv2.resize(img, (150, 150))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)

                self.list_widget_filter.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_filter.setValue(self.completed)
            self.progress_bar_filter.setValue(100)
        if self.radio_saltpeper.isChecked():
            self.filter_type = "SaltPeperNoise"
            for e in self.images_list:
                img = cv2.imread(e)
                img = Noise.noise_SaltPepper(img)
                self.filtered_images.append(img)
                img = cv2.resize(img, (150, 150))
                image = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)

                self.list_widget_filter.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_filter.setValue(self.completed)
            self.progress_bar_filter.setValue(100)

    def save_filter(self):
        img_folder = self.line_edit_filter.text()
        if os.path.exists(img_folder):
            for i in range(0, len(self.list_to_filter)):
                words = self.list_to_filter[i].split('\\')
                file_and_extension = words[len(words) - 1]
                file_and_extension_list = file_and_extension.split('.')
                file_name = file_and_extension_list[0]
                new_path = img_folder + '\\' + file_name + "-" + self.filter_type + "Filter.png"
                cv2.imwrite(new_path, self.filtered_images[i])
            self.label_save_filters.setText("Zapisano")
            self.label_save_filters.setStyleSheet('color: green')
        else:
            self.label_save_filters.setText("Nie znaleziono folderu")
            self.label_save_filters.setStyleSheet('color: red')

    def show_clicked_filter(self):
        row = self.list_widget_filter.currentRow()
        cv2.imshow("Image", self.filtered_images[row])
        cv2.waitKey()

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
            self.label_group_a.setText("Obrazy nie rozmyte")
            self.label_group_b.setText("Obrazy rozmyte")
            for e in self.images_list:
                if testBlurrLaplacianVariance.test_if_not_blurred(e):
                    self.images_group_a.append(e)

                    img = cv2.imread(e)
                    img = cv2.resize(img, (50, 50))
                    image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                                         QtGui.QImage.Format_RGB888).rgbSwapped()
                    icon = QtGui.QIcon()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    item = QListWidgetItem(self.getShortFilePath(e))
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
                    item = QListWidgetItem(self.getShortFilePath(e))
                    item.setIcon(icon)

                    self.list_widget_b.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_groups.setValue(self.completed)
            self.progress_bar_groups.setValue(100)
        if self.check_box_edge.isChecked():
            self.label_group_a.setText("Obrazy z dużą liczbą krawędzi")
            self.label_group_b.setText("Obrazy z małą liczbą krawędzi")
            model = testEdgesKMeans.read_model()
            for e in self.images_list:
                img = cv2.imread(e)
                if testEdgesKMeans.test_group(img, model) == 0:
                    self.images_group_a.append(e)

                    img = cv2.imread(e)
                    img = cv2.resize(img, (50, 50))
                    image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                                         QtGui.QImage.Format_RGB888).rgbSwapped()
                    icon = QtGui.QIcon()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    item = QListWidgetItem(self.getShortFilePath(e))
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
                    item = QListWidgetItem(self.getShortFilePath(e))
                    item.setIcon(icon)

                    self.list_widget_b.addItem(item)
                self.completed += int(100 / length)
                self.progress_bar_groups.setValue(self.completed)
            self.progress_bar_groups.setValue(100)

    def clear_divide(self):
        self.label_group_a.setText("Grupa pierwsza")
        self.label_group_b.setText("Grupa druga")
        self.images_divided = False
        self.progress_bar_groups.setValue(0)
        self.images_group_a.clear()
        self.images_group_b.clear()
        self.list_widget_a.clear()
        self.list_widget_b.clear()

    def show_clicked_a(self):
        row = self.list_widget_a.currentRow()
        cv2.imshow("Image", cv2.imread(self.images_group_a[row]))
        cv2.waitKey()

    def show_clicked_b(self):
        row = self.list_widget_b.currentRow()
        cv2.imshow("Image", cv2.imread(self.images_group_b[row]))
        cv2.waitKey()

    def segmentation(self):
        self.list_to_segment = []
        self.list_widget_regions.clear()
        self.list_widget_edges.clear()
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
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)
                self.list_widget_edges.addItem(item)
                self.edges_list.append(edges)

                image = QtGui.QImage(regions.data, regions.shape[1], regions.shape[0], regions.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)
                self.list_widget_regions.addItem(item)
                self.regions_list.append(regions)

                self.completed += int(100 / length)
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
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)
                self.list_widget_edges.addItem(item)
                self.edges_list.append(edges)
                image = QtGui.QImage(regions.data, regions.shape[1], regions.shape[0], 3 * regions.shape[1],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)
                self.list_widget_regions.addItem(item)
                self.regions_list.append(regions)

                self.completed += int(100 / length)
                self.progress_bar_seg.setValue(self.completed)
            self.progress_bar_seg.setValue(100)
        if self.check_box_region.isChecked():
            self.seg_type = "Region"
            for e in self.list_to_segment:
                image = cv2.imread(e, 1)
                regions, edges = Region.auto_region_growing(image)
                image = QtGui.QImage(edges.data, edges.shape[1], edges.shape[0], edges.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)
                self.list_widget_edges.addItem(item)
                self.edges_list.append(edges)
                image = QtGui.QImage(regions.data, regions.shape[1], regions.shape[0], regions.shape[1],
                                     QtGui.QImage.Format_Grayscale8)
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap.fromImage(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item = QListWidgetItem(self.getShortFilePath(e))
                item.setIcon(icon)
                self.list_widget_regions.addItem(item)
                self.regions_list.append(regions)

                self.completed += int(100 / length)
                self.progress_bar_seg.setValue(self.completed)
            self.progress_bar_seg.setValue(100)

    def clear_seg(self):
        self.list_widget_regions.clear()
        self.list_widget_edges.clear()
        self.edges_list.clear()
        self.regions_list.clear()
        self.progress_bar_seg.setValue(0)

    def show_clicked_edges(self):
        row = self.list_widget_edges.currentRow()
        cv2.imshow("Image", self.edges_list[row])
        cv2.waitKey()

    def show_clicked_regions(self):
        row = self.list_widget_regions.currentRow()
        cv2.imshow("Image", self.regions_list[row])
        cv2.waitKey()

    def save_seg(self):
        img_folder = self.text_save_location.text()
        if os.path.exists(img_folder):
            for i in range(0, len(self.list_to_segment)):
                words = self.list_to_segment[i].split('\\')
                file_and_extension = words[len(words) - 1]
                file_and_extension_list = file_and_extension.split('.')
                file_name = file_and_extension_list[0]
                extension_name = file_and_extension_list[1]
                binary_path = img_folder + '\\' + file_name + "-" + self.seg_type + "Binary." + extension_name
                marker_path = img_folder + '\\' + file_name + "-" + self.seg_type + "Marker.png"
                cv2.imwrite(binary_path, self.edges_list[i])
                plt.imsave(marker_path, self.regions_list[i])
            self.label_save_seg.setText("Zapisano")
            self.label_save_seg.setStyleSheet('color: green')
        else:
            self.label_save_seg.setText("Nie znaleziono folderu")
            self.label_save_seg.setStyleSheet('color: red')


    def evaluation(self):
        self.table_ev_results.setRowCount(0)
        self.completed = 0
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
        true_segmentation_paths = self.loadImagesPathsInFolder(ground_true_folder, [".bmp"])
        predicted_folder = self.line_ev_pred_folder.text()
        predicted_segmentation_paths = self.loadImagesPathsInFolder(predicted_folder, [".jpg", ".png", ".bmp"])
        predicted_segmentation_paths = self.filterPredictedPathsForEvaluation(predicted_segmentation_paths, 'Binary')
        evaluation_paths = self.connectPaths(predicted_segmentation_paths, true_segmentation_paths)
        self.progressBar_evaluation.setValue(0)
        self.evaluation_results = []
        length = len(evaluation_paths)
        row = 0
        # ev = evaluation.Evaluation(evaluation_paths, algorithms)
        # for current_results in ev.getResults():
        for paths in evaluation_paths:
            paths_nested = []
            paths_nested.append(paths)
            ev = evaluation.Evaluation(paths_nested, algorithms)
            # Evaluation
            current_results = ev.getResults()
            # print("evaluation results: ", evaluation_results)
            # Displaying result
            current_results = current_results[0]
            pred_img = cv2.imread(current_results["predicted path"])
            true_img = cv2.imread(current_results["true path"])
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
            # rowPosition = self.table_ev_results.rowCount()
            self.table_ev_results.setRowCount(length)
            item = QTableWidgetItem(self.getShortFilePath(current_results["predicted path"]))
            item.setIcon(pred_icon)
            self.table_ev_results.setItem(row, 0, item)
            item = QTableWidgetItem(self.getShortFilePath(current_results["true path"]))
            item.setIcon(true_icon)
            self.table_ev_results.setItem(row, 1, item)
            self.table_ev_results.setItem(row, 2, QTableWidgetItem(str(round(current_results["jaccarda index weighted"],
                                                                             4))))
            self.table_ev_results.setItem(row, 3, QTableWidgetItem(str(round(current_results["f1 score weighted"], 4))))
            self.table_ev_results.setItem(row, 4, QTableWidgetItem(str(round(current_results["explained variance score"]
                                                                             , 4))))
            self.table_ev_results.setItem(row, 5, QTableWidgetItem(str(round(current_results["mean squared error"],
                                                                             4))))
            self.table_ev_results.resizeRowsToContents()
            row += 1
            # item = QListWidgetItem(pred_icon, true_icon, evaluation_results[0]["predicted path"],
            # evaluation_results[0] ["jaccarda index weighted"], evaluation_results[0]["f1 score weighted"],
            # evaluation_results[0] ["explained variance score"], evaluation_results[0]["mean squared error"]) #
            # item.setIcon(icon) self.list_ev_results.addItem(item)

            self.evaluation_results.append(current_results)
            # Update progress bar
            self.completed += int(100 / length)
            self.progressBar_evaluation.setValue(self.completed)
        self.progressBar_evaluation.setValue(100)
        # TODO displaying results

    def saveEvaluationResults(self):
        self.progressBar_evaluation.setValue(0)
        group = self.combo_groups.currentIndex()
        clear = self.combo_filter.currentIndex()
        noise = self.combo_noise.currentIndex()
        segmentation = self.combo_seg.currentIndex()
        path = self.lineEdit_save_ev.text()
        if group == 4:
            path += "-AllImages"
        elif group == 0:
            path += "-EdgesMuch"
        elif group == 1:
            path += "-EdgesLittle"
        elif group == 2:
            path += "-Blurr"
        elif group == 3:
            path += "-NotBlurr"
        if noise == 3:
            path += "-NoNoise"
        elif noise == 0:
            path += "-Gaussian"
        elif noise == 1:
            path += "-SaltPeper"
        elif noise == 2:
            path += "-Speckle"
        if clear == 3:
            path += "-NoClear"
        elif clear == 0:
            path += "-Gaussian"
        elif clear == 1:
            path += "-Bilateral"
        elif clear == 2:
            path += "-Median"
        if segmentation == 3:
            path += "-NoSegmentation"
        elif segmentation == 0:
            path += "-EdgesBased"
        elif segmentation == 1:
            path += "-RegionBased"
        elif segmentation == 2:
            path += "-Watershed"
        path += '.csv'
        clearedResults = self.evaluation_results
        length = len(clearedResults)
        with open(path, mode='w') as result_file:
            fieldnames = ['image name', 'jaccarda index weighted', 'f1 score weighted', 'explained variance score',
                          'mean squared error']
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            writer.writeheader()
            for dict in clearedResults:
                path = dict['true path']
                name = self.getFileNameFromFilePath(path)
                dict['image name'] = name
                del dict['predicted image']
                del dict['predicted path']
                del dict['true image']
                del dict['true path']
                del dict['jaccarda index micro']
                del dict['jaccarda index macro']
                del dict['f1 score micro']
                del dict['f1 score macro']
                writer.writerow(dict)
                # Update progress bar
                self.completed += int(100 / length)
                self.progressBar_evaluation.setValue(self.completed)
        self.progressBar_evaluation.setValue(100)
        # with open(path, mode='w') as result_file:
        #     fieldnames = ['image name', 'jaccarda index weighted', 'f1 score weighted', 'explained variance score',
        #                   'mean squared error']
        #     writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        #     writer.writeheader()
        #     for dict in clearedResults:
        #         writer.writerow(dict)

    def filterPredictedPathsForEvaluation(self, paths, key_string):
        filtered_paths = [path for path in paths if key_string in path]
        return filtered_paths

    def connectPaths(self, predicted, true):
        connected_paths = []
        for pred_path in predicted:
            file_name = self.getFileNameFromFilePath(pred_path)
            image_name = self.getImageNameFromFileName(file_name)
            true_path = self.findGroundTruthForImage(image_name, true)
            # TODO exception not found
            if true_path == "":
                None
            # TODO
            connected_paths.append([true_path, pred_path])
        return connected_paths

    def getFileNameFromFilePath(self, filePath):
        words = filePath.split('\\')
        fileAndExtension = words[len(words) - 1]
        fileAndExtensionList = fileAndExtension.split('.')
        fileName = fileAndExtensionList[0]
        extensionName = fileAndExtensionList[1]
        return fileName

    def getImageNameFromFileName(self, fileName):
        words = fileName.split('-')
        image = words[0]
        return image

    def findGroundTruthForImage(self, imageName, groundTruthPaths):
        for path in groundTruthPaths:
            fileName = self.getFileNameFromFilePath(path)
            foundName = (fileName.split('-'))[0]
            if foundName == imageName:
                return path
        return ""

    def getShortFilePath(self, path):
        hierarchy = path.split('\\')
        length = len(hierarchy)
        if length < 3:
            return path
        else:
            short_path = hierarchy[length-3] + "\\" + hierarchy[length-2] + "\\" + hierarchy[length-1]
            return short_path

    def loadImagesPathsInFolder(self, path, valid_images):
        path_list = []
        # TODO exception
        if os.path.exists(path):
            for f in os.listdir(path):
                ext = os.path.splitext(f)[1]
                if ext.lower() in valid_images:
                    f = os.path.join(path, f)
                    path_list.append(f)
        return path_list


def main():
    app = QApplication([])
    vp = MainClass()
    vp.show()
    app.exec_()


if __name__ == '__main__':
    main()
