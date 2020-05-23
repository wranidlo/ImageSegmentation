import numpy as np
import cv2
from matplotlib import pyplot as plt

class watershedSegmentation:
	def __init__(self, img_path, res_folder_path="results"):
		self.originalPath = img_path
		self.resultsFolderPath = res_folder_path + "\\"
		self.originalImage = self.loadImage()
		self.grayscaleImage, self.thresholdValue, self.thresholdedImage = self.thresholdImage()
		self.kernel, self.opening = self.noiseRemoval()
		self.sure_bg = self.findSureBackgournd(self.opening, self.kernel)
		self.sure_fg = self.findSureForeground(self.opening)
		self.unknown = self.findUnknownRegion()
		self.contours, self.hierarchy = self.findContours()
		self.marker = self.markerCreateAndLabeling()
		self.watershed()
		self.markerImage = self.convertMarkerToImage()

	def loadImage(self):
		return cv2.imread(self.originalPath) # Reading the image

	def thresholdImage(self):
		gray = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY)  # Conversion to gray scale
		ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # thresholding
		return gray, ret, thresh

	def noiseRemoval(self):
		# noise removal
		kernel = np.ones((3, 3), np.uint8)
		opening = cv2.morphologyEx(self.thresholdedImage, cv2.MORPH_OPEN, kernel, iterations=2)
		return kernel, opening

	def findSureBackgournd(self, opening, kernel):
		# sure background area
		sure_bg = cv2.dilate(opening, kernel, iterations=3)
		return sure_bg

	def findSureForeground(self, opening):
		# Finding sure foreground area
		dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
		self.thresholdValue, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
		return sure_fg

	def findUnknownRegion(self):
		# Finding unknown region
		self.sure_fg = np.uint8(self.sure_fg)
		unknown = cv2.subtract(self.sure_bg, self.sure_fg)
		return unknown

	def findContours(self):
		contours, hierarchy = cv2.findContours(self.sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		return contours, hierarchy

	def markerCreateAndLabeling(self):
		# Creating a numpy array for markers and converting the image to 32 bit using dtype paramter
		marker = np.zeros((self.grayscaleImage.shape[0], self.grayscaleImage.shape[1]), dtype=np.int32)
		marker = np.int32(self.sure_fg) + np.int32(self.sure_bg)
		# Marker Labelling
		for id in range(len(self.contours)):
			cv2.drawContours(marker, self.contours, id, id + 2, -1)
		marker = marker + 1
		marker[self.unknown == 255] = 0
		return marker

	def watershed(self):
		cv2.watershed(self.originalImage, self.marker)
		self.originalImage[self.marker == -1] = (0, 0, 255)

	def convertMarkerToImage(self):
		# a colormap and a normalization instance for saving marker
		cmap = plt.cm.jet
		norm = plt.Normalize(vmin=self.marker.min(), vmax=self.marker.max())
		# map the normalized data to colors
		# image is now RGBA (512x512x4)
		return cmap(norm(self.marker))

	def getResults(self):
		#binary, color
		return self.unknown, self.markerImage

	def saveResults(self):
		words = self.originalPath.split('/')
		fileAndExtension = words[len(words)-1]
		fileAndExtensionList = fileAndExtension.split('.')
		fileName = fileAndExtensionList[0]
		extensionName = fileAndExtensionList[1]
		binaryPath = self.resultsFolderPath + fileName + "-WatershedBinary." + extensionName
		markerPath = self.resultsFolderPath + fileName + "-WatershedMarker.png"
		cv2.imwrite(binaryPath, self.unknown)
		plt.imsave(markerPath, self.markerImage)
		# cv2.imwrite('results/WSBInary.jpg', self.unknown)
		# plt.imsave('results/WSMarker.png', self.markerImage)


if __name__ == "__main__":
	segmentation = watershedSegmentation('images/test/3096.jpg')
	segmentation.saveResults()