import numpy as np
from sklearn.metrics import jaccard_similarity_score, jaccard_score
import cv2
from matplotlib import pyplot as plt

#Jaccard Index also known as Intersection-Over-Union (IoU)
print("Give path to the ground truth image")
path_true = input()
print("Give path to the image after segmentation")
path_predict = input()
#img_true = cv2.imread('ground truth/gray/human/3096.bmp')
#img_pred = cv2.imread('results/WSBinary.jpg')
img_true = cv2.imread(path_true)
img_pred = cv2.imread(path_predict)
img_true=np.array(img_true).ravel()
img_pred=np.array(img_pred).ravel()
#iou = jaccard_similarity_score(img_true, img_pred)
iou_none_avg = jaccard_score(img_true, img_pred, average=None)
iou_micro_avg = jaccard_score(img_true, img_pred, average="micro")
iou_macro_avg = jaccard_score(img_true, img_pred, average="macro")
iou_weighted_avg = jaccard_score(img_true, img_pred, average="weighted")
print("Jaccarda Index evaluation resault for different types of average:")
print("none: ", iou_none_avg)
print("macro: ", iou_macro_avg)
print("micro: ", iou_micro_avg)
print("weighted: ", iou_weighted_avg)
