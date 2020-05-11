import numpy as np
from sklearn.metrics import jaccard_similarity_score, jaccard_score, f1_score, recall_score, mean_squared_error, explained_variance_score
import cv2
from matplotlib import pyplot as plt


def printMainMenu():
    print("\nMENU")
    print("1. load images to evaluation")
    print("2. delete images from segmentation")
    print("3. display loaded images")
    print("4. display current evalutaion results")
    print("5. evaluate loaded segmentation with all algorithms")
    print("6. display algorithms of evaluation")
    if preconfiguredPaths == 0:
        print("7. toggle on preconfigured paths (TESTING)")
    else:
        print("7. toggle off preconfigured paths (TESTING)")
    print("0. exit")


def printAlgorithmsMenu():
    # classification
    print("1. Jaccarda Index")
    print("2. F1 Score")
    # regression
    print("3. Explained Varince Score (additional)")
    print("4. Mean Squared Error")
    print("0. go back")


class Evaluation:
    def __init__(self):
        self.trueSeg = []
        self.predSeg = []
        self.trueSegPaths = []
        self.predSegPaths = []
        self.js_none = []
        self.js = []
        self.f1_none = []
        self.f1 = []
        self.evs = []
        self.mse = []

    def loadImagesSet(self, pathPred, pathTrue):
        self.loadTrueImage(pathTrue)
        self.loadPredictedImage(pathPred)
        self.trueSegPaths.append(pathTrue)
        self.predSegPaths.append(pathPred)

    def deleteImageSet(self, id):
        #TODO exception for bad ID
        id = int(id)
        del self.trueSeg[id]
        del self.predSeg[id]
        del self.trueSegPaths[id]
        del self.predSegPaths[id]
        if id > len(self.js_none):
            del self.js_none[id]
            del self.js[id]
            del self.f1_none[id]
            del self.f1[id]
            del self.evs[id]
            del self.mse[id]

    def displayImages(self):
        id = 0;
        for path in self.trueSegPaths:
            print("--------------------------------------------------")
            print("ID:", id)
            print("true: " + path)
            print("pred: " + self.predSegPaths[id])
            id += 1
        print("--------------------------------------------------")

    def loadImageAsArray(self, path):
        image = cv2.imread(path)
        return np.array(image).ravel()

    def loadTrueImage(self, path):
        self.trueSeg.append(self.loadImageAsArray(path))

    def loadPredictedImage(self, path):
        self.predSeg.append(self.loadImageAsArray(path))

    # Jaccard Index also known as Intersection-Over-Union (IoU)
    def jaccardScoreAlgorithm(self):
        # iou = jaccard_similarity_score(self.img_true, self.img_pred) - old method
        id = 0
        for pred in self.predSeg:
            self.js_none.append(jaccard_score(self.trueSeg[id], pred, average=None))
            self.js.append({})
            self.js[id]['micro'] = jaccard_score(self.trueSeg[id], pred, average="micro")
            self.js[id]['macro'] = jaccard_score(self.trueSeg[id], pred, average="macro")
            self.js[id]['weighted'] = jaccard_score(self.trueSeg[id], pred, average="weighted")
            id += 1

    # F1 Score also known as Dice Coefficient
    def F1ScoreAlgorithm(self):
        id = 0
        for pred in self.predSeg:
            self.f1_none.append(f1_score(self.trueSeg[id], pred, average=None))
            self.f1.append({})
            self.f1[id]['micro'] = f1_score(self.trueSeg[id], pred, average="micro")
            self.f1[id]['macro'] = f1_score(self.trueSeg[id], pred, average="macro")
            self.f1[id]['weighted'] = f1_score(self.trueSeg[id], pred, average="weighted")
            id += 1

    # Explained Variance Score
    def explainedVarianceScoreAlgorithm(self):
        id = 0
        for pred in self.predSeg:
            self.evs.append(explained_variance_score(self.trueSeg[id], pred))
            id += 1

    # Mean absolute error
    def meanSquaredErrorAlgorithm(self):
        id = 0
        for pred in self.predSeg:
            self.mse.append(mean_squared_error(self.trueSeg[id], pred))
            id += 1

    # Call all evaluation algorithms
    def allAlgorithms(self):
        self.jaccardScoreAlgorithm()
        self.F1ScoreAlgorithm()
        self.explainedVarianceScoreAlgorithm()
        self.meanSquaredErrorAlgorithm()

    def displayResults(self):
        print("Results:")
        id = 0
        for _ in self.js:
            print("--------------------------------------------------")
            print("Evaluation for images", id)
            print("\nJaccarda Index (average type: result [0-1] )")
            print("None: ", self.js_none[id])
            for k in self.js[id]:
                print(k, ": ", self.js[id][k])
            print("\nF1 Score (average type: result [0-1] )")
            print("None: ", self.f1_none[id])
            for k in self.f1[id]:
                print(k, ": ", self.f1[id][k])
            print("\nExplained Variance Score", self.evs[id])
            print("\nMean squared error: ", self.mse[id])
            id += 1
        print("--------------------------------------------------")


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
                pathTrue = input("Give path to ground truth image:")
                pathPred = input("Give path to predicted image:")
                evaluation.loadImagesSet(pathPred, pathTrue)
                print("\nimages loaded from:\n" + pathTrue + "\n\tand\n" + pathPred)
            else:
                evaluation.loadImagesSet(pathPredSeg, pathTrueSeg)
                print("\nimages loaded from:\n" + pathTrueSeg + "\n\tand\n" + pathPredSeg)
        elif option == 2:
            id = input("Give id images to delete")
            evaluation.deleteImageSet(id)
        elif option == 3:
            evaluation.displayImages()
        elif option == 4:
            evaluation.displayResults()
        elif option == 5:
            evaluation.allAlgorithms()
            print("\nevaluation ended successfully")
        elif option == 6:
            printAlgorithmsMenu()
            option_2 = int(input())
            # if option_2 == 1:
            #     evaluation.jaccardScoreAlgorithm()
            #     print("\nevaluation ended successfully")
            # elif option_2 == 2:
            #     evaluation.F1ScoreAlgorithm()
            #     print("\nevaluation ended successfully")
            # elif option_2 == 3:
            #     evaluation.explainedVarianceScoreAlgorithm()
            #     print("\nevaluation ended successfully")
            # elif option_2 == 4:
            #     evaluation.meanSquaredErrorAlgorithm()
            #     print("\nevaluation ended successfully")
        elif option == 7:
            if preconfiguredPaths == 0:
                preconfiguredPaths = 1
                print("\npreconfigured paths: on")
            else:
                preconfiguredPaths = 0
                print("\npreconfigured paths: off")