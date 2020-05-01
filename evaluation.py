import numpy as np
from sklearn.metrics import jaccard_similarity_score, jaccard_score, f1_score, recall_score, mean_absolute_error
import cv2
from matplotlib import pyplot as plt


def printMainMenu():
    print("MENU")
    print("1. load true segmentation")
    print("2. load predicted segmentation")
    print("3. display current evalutaion results")
    print("4. evaluate loaded segmentation with all algorithms")
    print("5. choose algorithm to evaluate loaded segmentation")
    if preconfiguredPaths == 0:
        print("6. toggle on preconfigured paths (TESTING)")
    else:
        print("6. toggle off preconfigured paths (TESTING)")
    print("0. exit")


def printAlgorithmsMenu():
    print("1. Jaccarda Score")
    print("2. F1 Score")
    print("3. Recall Score (additional)")
    print("4. Mean absolute error (additional)")
    print("0. go back")


class Evaluation:
    def __init__(self):
        self.img_true = None
        self.img_pred = None
        self.js_none = None
        self.js = {}
        self.f1_none = None
        self.f1 = {}
        # additional algorithms
        self.rc = {}
        self.mae = None

    def loadImageAsArray(self, path):
        image = cv2.imread(path)
        return np.array(image).ravel()

    def setTrueImage(self, path):
        self.img_true = self.loadImageAsArray(path)

    def setPredictedImage(self, path):
        self.img_pred = self.loadImageAsArray(path)

    # Jaccard Index also known as Intersection-Over-Union (IoU)
    def jaccardScoreAlgorithm(self):
        # iou = jaccard_similarity_score(self.img_true, self.img_pred) - old method
        self.js_none = jaccard_score(self.img_true, self.img_pred, average=None)
        self.js['micro'] = jaccard_score(self.img_true, self.img_pred, average="micro")
        self.js['macro'] = jaccard_score(self.img_true, self.img_pred, average="macro")
        self.js['weighted'] = jaccard_score(self.img_true, self.img_pred, average="weighted")

    # F1 Score also known as Dice Coefficient
    def F1ScoreAlgorithm(self):
        self.f1_none = f1_score(self.img_true, self.img_pred, average=None)
        self.f1['micro'] = f1_score(self.img_true, self.img_pred, average="micro")
        self.f1['macro'] = f1_score(self.img_true, self.img_pred, average="macro")
        self.f1['weighted'] = f1_score(self.img_true, self.img_pred, average="weighted")

    # Recall Score
    def recallScoreAlgorithm(self):
        self.rc['weighted'] = recall_score(self.img_true, self.img_pred, average="weighted")

    # Mean absolute error
    def meanAbsoluteErrorAlgorithm(self):
        self.mae = mean_absolute_error(self.img_true, self.img_pred)

    # Call all evaluation algorithms
    def allAlgorithms(self):
        self.jaccardScoreAlgorithm()
        self.F1ScoreAlgorithm()
        self.recallScoreAlgorithm()
        self.meanAbsoluteErrorAlgorithm()

    def displayResults(self):
        print("Results:")
        print("\nJaccarda Index (average type: result [0-1] )")
        print("None: ", self.js_none)
        for k in self.js:
            print(k, ": ", self.js[k])
        print("\nF1 Score (average type: result [0-1] )")
        print("None: ", self.f1_none)
        for k in self.f1:
            print(k, ": ", self.f1[k])
        print("\nadditional algorithms")
        print("\nRecall Score (average type: result [0-1] )")
        for k in self.rc:
            print(k, ": ", self.rc[k])
        print("\nMean absolute error: ", self.mae)


if __name__ == "__main__":
#     evaluation = Evaluation()
#     evaluation.setTrueImage('ground truth/gray/human/3096.bmp')
#     evaluation.setPredictedImage('results/WSBinary.jpg')
#     evaluation.allAlgorithms()
#     evaluation.displayResults()
# else:
    pathTrueSeg = 'ground truth/gray/human/3096.bmp'
    pathPredSeg = 'results/WSBinary.jpg'
    preconfiguredPaths = 0
    evaluation = Evaluation()
    while True:
        printMainMenu()
        option = int(input())
        if option == 0:
            break
        elif option == 1:
            if preconfiguredPaths == 0:
                path = input("Give path to image:")
                evaluation.setTrueImage(path)
            else:
                evaluation.setTrueImage(pathTrueSeg)
        elif option == 2:
            if preconfiguredPaths == 0:
                path = input("Give path to image:")
                evaluation.setPredictedImage(path)
            else:
                evaluation.setPredictedImage(pathPredSeg)
        elif option == 3:
            evaluation.displayResults()
        elif option == 4:
            evaluation.allAlgorithms()
        elif option == 5:
            printAlgorithmsMenu()
            option_2 = int(input())
            if option_2 == 1:
                evaluation.jaccardScoreAlgorithm()
            elif option_2 == 2:
                evaluation.F1ScoreAlgorithm()
            elif option_2 == 3:
                evaluation.recallScoreAlgorithm()
            elif option_2 == 4:
                evaluation.meanAbsoluteErrorAlgorithm()
        elif option == 6:
            if preconfiguredPaths == 0:
                preconfiguredPaths = 1
            else:
                preconfiguredPaths = 0