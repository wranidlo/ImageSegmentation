import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn import cluster
import pickle


def get_edges_transformed(img):
    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges


def get_edges(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def get_edges_value(edges):
    x = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
    return x


def learn():
    image_list = []
    for filename in glob.glob('D:\\Users\\baryl\Documents\GitHub\ImageSegmentation\images\\train\\*.jpg'):
        img = cv2.imread(filename)
        img_edges_transformed = get_edges_transformed(img)
        img_edges = get_edges(img)
        image_list.append([get_edges_value(img_edges_transformed), get_edges_value(img_edges)])

    x = np.array(image_list)
    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(x)
    return k_means


def read_model():
    with open("model_edges.pkl", 'rb') as file:
        model = pickle.load(file)
    return model


def test_group(img, model):

    edges = get_edges_transformed(img)
    edges_transformed = get_edges_transformed(img)
    return model.predict([[get_edges_value(edges_transformed), get_edges_value(edges)]])


def main():
    # model = learn()
    # with open(pkl_filename, 'wb') as file:
    #    pickle.dump(model, file)
    model = read_model()

    test_names = ["D:\\Users\\baryl\Documents\GitHub\ImageSegmentation\images\\test\\3096.jpg",
                  "D:\\Users\\baryl\Documents\GitHub\ImageSegmentation\images\\test\\8023.jpg",
                  "D:\\Users\\baryl\Documents\GitHub\ImageSegmentation\images\\test\\12084.jpg",
                  "D:\\Users\\baryl\Documents\GitHub\ImageSegmentation\images\\test\\14037.jpg",
                  "D:\\Users\\baryl\Documents\GitHub\ImageSegmentation\images\\test\\16077.jpg"]
    test_list = []
    for e in test_names:
        im = cv2.imread(e)
        edges = get_edges_transformed(im)
        edges_transformed = get_edges_transformed(im)
        test_list.append([get_edges_value(edges_transformed), get_edges_value(edges)])

    print(model.predict(test_list))
    # first category is good for edge based segmentation


if __name__ == "__main__":
    main()
