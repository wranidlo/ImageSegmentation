import numpy as np
from sklearn.metrics import jaccard_similarity_score, jaccard_score, f1_score, recall_score, mean_absolute_error
import cv2
from matplotlib import pyplot as plt

#Preparing data
# print("Give path to the ground truth image")
# path_true = input()
# print("Give path to the image after segmentation")
# path_predict = input()
img_true = cv2.imread('ground truth/gray/human/3096.bmp')
img_pred = cv2.imread('results/WSBinary.jpg')
# img_true = cv2.imread(path_true)
# img_pred = cv2.imread(path_predict)
img_true=np.array(img_true).ravel()
img_pred=np.array(img_pred).ravel()
#Jaccard Index also known as Intersection-Over-Union (IoU)
#iou = jaccard_similarity_score(img_true, img_pred) - old method
iou_none_avg = jaccard_score(img_true, img_pred, average=None)
iou_micro_avg = jaccard_score(img_true, img_pred, average="micro")
iou_macro_avg = jaccard_score(img_true, img_pred, average="macro")
iou_weighted_avg = jaccard_score(img_true, img_pred, average="weighted")
print("Jaccarda Index evaluation resault for different types of average:")
#F1 Score also known as Dice Coefficient
f1_none_avg = f1_score(img_true, img_pred, average=None)
f1_micro_avg = f1_score(img_true, img_pred, average="micro")
f1_macro_avg = f1_score(img_true, img_pred, average="macro")
f1_weighted_avg = f1_score(img_true, img_pred, average="weighted")
#Displaying results
print("\nJaccarda Index evaluation resault for different types of average:")
print("none: ", iou_none_avg)
print("macro: ", iou_macro_avg)
print("micro: ", iou_micro_avg)
print("weighted: ", iou_weighted_avg)
print("\nF1 Score evaluation resault for different types of average:")
print("none: ", iou_none_avg)
print("macro: ", iou_macro_avg)
print("micro: ", iou_micro_avg)
print("weighted: ", iou_weighted_avg)
#Other methods
print("Other algorithms")
rc_weighted_avg = recall_score(img_true, img_pred, average="weighted")
print("Recall_score: ", rc_weighted_avg)
mae = mean_absolute_error(img_true, img_pred)
print("Mean absolute error: ", mae)