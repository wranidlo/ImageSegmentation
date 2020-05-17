import numpy as np
from sklearn.metrics import jaccard_similarity_score, jaccard_score, f1_score, recall_score, mean_squared_error, explained_variance_score
import cv2
from matplotlib import pyplot as plt

# TODO do something with that
# def printMainMenu():
#     print("\nMENU")
#     print("1. load images to evaluation")
#     print("2. delete images from segmentation")
#     print("3. display loaded images")
#     print("4. display current evalutaion results")
#     print("5. evaluate loaded segmentation with all algorithms")
#     print("6. display algorithms of evaluation")
#     if preconfiguredPaths == 0:
#         print("7. toggle on preconfigured paths (TESTING)")
#     else:
#         print("7. toggle off preconfigured paths (TESTING)")
#     print("0. exit")
#
#
# def printAlgorithmsMenu():
#     # classification
#     print("1. Jaccarda Index")
#     print("2. F1 Score")
#     # regression
#     print("3. Explained Varince Score (additional)")
#     print("4. Mean Squared Error")
#     print("0. go back")


class Evaluation:
    # IMPORTANT! seg_path is list of lists with truePath at 0 and predPath at 1
    def __init__(self, seg_paths=[], algorithms=[0, 0, 0, 0]):
        self.segPaths = seg_paths
        self.results = []
        self.loadImagesFromPaths()
        self.algorithmsToExecute = algorithms
        if self.algorithmsToExecute[0] == 1 and self.algorithmsToExecute[1] == 1 and self.algorithmsToExecute[2] == 1 \
                and self.algorithmsToExecute[3] == 1:
            self.allAlgorithms()
        if self.algorithmsToExecute[0] == 1:
            self.jaccardScoreAlgorithm()
        if self.algorithmsToExecute[1] == 1:
            self.F1ScoreAlgorithm()
        if self.algorithmsToExecute[2] == 1:
            self.explainedVarianceScoreAlgorithm()
        if self.algorithmsToExecute[3] == 1:
            self.meanSquaredErrorAlgorithm()

    def loadImagesFromPaths(self):
        for paths in self.segPaths:
            dict = {}
            true_dict = self.loadImageFromPath(paths[0])
            pred_dict = self.loadImageFromPath(paths[1])
            dict["predicted path"] = pred_dict["path"]
            dict["predicted image"] = pred_dict["image"]
            dict["true path"] = true_dict["path"]
            dict["true image"] = true_dict["image"]
            dict["jaccarda index micro"] = 0.0
            dict["jaccarda index macro"] = 0.0
            dict["jaccarda index weighted"] = 0.0
            dict["f1 score micro"] = 0.0
            dict["f1 score macro"] = 0.0
            dict["f1 score weighted"] = 0.0
            dict["explained variance score"] = 0.0
            dict["mean squared error"] = 0.0
            self.results.append(dict)

    def loadImageFromPath(self, path):
        dict = {}
        dict["path"] = path
        dict["image"] = self.loadImageAsArray(path)
        return dict

    def loadImageAsArray(self, path):
        image = cv2.imread(path)
        return np.array(image).ravel()

    def loadImageDict(self, seg_paths):
        dict = {}
        true_dict = self.loadImageFromPath(seg_paths[0])
        pred_dict = self.loadImageFromPath(seg_paths[1])
        dict["predicted path"] = pred_dict["path"]
        dict["predicted image"] = pred_dict["image"]
        dict["true path"] = true_dict["path"]
        dict["true image"] = true_dict["image"]
        dict["jaccarda index micro"] = 0.0
        dict["jaccarda index macro"] = 0.0
        dict["jaccarda index weighted"] = 0.0
        dict["f1 score micro"] = 0.0
        dict["f1 score macro"] = 0.0
        dict["f1 score weighted"] = 0.0
        dict["explained variance score"] = 0.0
        dict["mean squared error"] = 0.0
        self.results.append(dict)

    def deleteImageDict(self, id):
        #TODO exception for bad ID
        id = int(id)
        del self.results[id]

    # Jaccard Index also known as Intersection-Over-Union (IoU)
    def jaccardScoreAlgorithm(self):
        # iou = jaccard_similarity_score(self.img_true, self.img_pred) - old method
        for id in range(0, len(self.results)):
            true = self.results[id]["true image"]
            pred = self.results[id]["predicted image"]
            self.results[id]["jaccarda index micro"] = jaccard_score(true, pred, average="micro")
            self.results[id]["jaccarda index macro"] = jaccard_score(true, pred, average="macro")
            self.results[id]["jaccarda index weighted"] = jaccard_score(true, pred, average="weighted")

    # F1 Score also known as Dice Coefficient
    def F1ScoreAlgorithm(self):
        for id in range(0, len(self.results)):
            true = self.results[id]["true image"]
            pred = self.results[id]["predicted image"]
            self.results[id]["f1 score micro"] = f1_score(true, pred, average="micro")
            self.results[id]["f1 score macro"] = f1_score(true, pred, average="macro")
            self.results[id]["f1 score weighted"] = f1_score(true, pred, average="weighted")

    # Explained Variance Score
    def explainedVarianceScoreAlgorithm(self):
        for id in range(0, len(self.results)):
            true = self.results[id]["true image"]
            pred = self.results[id]["predicted image"]
            self.results[id]["explained variance score"] = explained_variance_score(true, pred)

    # Mean absolute error
    def meanSquaredErrorAlgorithm(self):
        for id in range(0, len(self.results)):
            true = self.results[id]["true image"]
            pred = self.results[id]["predicted image"]
            self.results[id]["mean squared error"] = mean_squared_error(true, pred)

    # Call all evaluation algorithms
    def allAlgorithms(self):
        self.jaccardScoreAlgorithm()
        self.F1ScoreAlgorithm()
        self.explainedVarianceScoreAlgorithm()
        self.meanSquaredErrorAlgorithm()

    def getResults(self):
        return self.results

    def displayResults(self):
        print("Results:")
        for dict in self.results:
            print("--------------------------------------------------")
            print("True path: ", dict["true path"])
            print("Predicted path: ", dict["predicted path"])
            print("Jaccarda Index (average type: result [0-1]):")
            print("micro: ", dict["jaccarda index micro"])
            print("macro: ", dict["jaccarda index macro"])
            print("weighted: ", dict["jaccarda index weighted"])
            print("F1 Score (average type: result [0-1]):")
            print("micro: ", dict["f1 score micro"])
            print("macro: ", dict["f1 score macro"])
            print("weighted: ", dict["f1 score weighted"])
            print("Explained Variance Score", dict["explained variance score"])
            print("Mean squared error: ", dict["mean squared error"])
        print("--------------------------------------------------")


if __name__ == "__main__":
    # evaluation = Evaluation()
    # evaluation.setTrueImage('ground truth/gray/human/3096.bmp')
    # evaluation.setPredictedImage('results/WSBInary.jpg')
    # evaluation.allAlgorithms()
    # evaluation.displayResults()
    list = []
    list.append(["ground truth/gray/human/3096.bmp", "results/3096-WatershedBinary.jpg"])
    evaluation = Evaluation(list)
    evaluation.allAlgorithms()
    evaluation.displayResults()

    # TODO do something with that
    # pathTrueSeg = 'ground truth/gray/human/3096.bmp'
    # pathPredSeg = 'results/WSBInary.jpg'
    # preconfiguredPaths = 0
    # evaluation = Evaluation()
    # while True:
    #     printMainMenu()
    #     option = int(input())
    #     if option == 0:
    #         break
    #     elif option == 1:
    #         if preconfiguredPaths == 0:
    #             pathTrue = input("Give path to ground truth image:")
    #             pathPred = input("Give path to predicted image:")
    #             evaluation.loadImagesSet(pathPred, pathTrue)
    #             print("\nimages loaded from:\n" + pathTrue + "\n\tand\n" + pathPred)
    #         else:
    #             evaluation.loadImagesSet(pathPredSeg, pathTrueSeg)
    #             print("\nimages loaded from:\n" + pathTrueSeg + "\n\tand\n" + pathPredSeg)
    #     elif option == 2:
    #         id = input("Give id images to delete")
    #         evaluation.deleteImageSet(id)
    #     elif option == 3:
    #         evaluation.displayImages()
    #     elif option == 4:
    #         evaluation.displayResults()
    #     elif option == 5:
    #         evaluation.allAlgorithms()
    #         print("\nevaluation ended successfully")
    #     elif option == 6:
    #         printAlgorithmsMenu()
    #         option_2 = int(input())
    #         # if option_2 == 1:
    #         #     evaluation.jaccardScoreAlgorithm()
    #         #     print("\nevaluation ended successfully")
    #         # elif option_2 == 2:
    #         #     evaluation.F1ScoreAlgorithm()
    #         #     print("\nevaluation ended successfully")
    #         # elif option_2 == 3:
    #         #     evaluation.explainedVarianceScoreAlgorithm()
    #         #     print("\nevaluation ended successfully")
    #         # elif option_2 == 4:
    #         #     evaluation.meanSquaredErrorAlgorithm()
    #         #     print("\nevaluation ended successfully")
    #     elif option == 7:
    #         if preconfiguredPaths == 0:
    #             preconfiguredPaths = 1
    #             print("\npreconfigured paths: on")
    #         else:
    #             preconfiguredPaths = 0
    #             print("\npreconfigured paths: off")