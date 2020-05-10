import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_edges(img):
    edges = cv2.Canny(img, 100, 200)

    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    return edges


def get_shape_from_edges(edges):

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    im_filled = cv2.dilate(edges, kernel, iterations=1)
    im_filled = cv2.erode(im_filled, kernel, iterations=1)
    h, w = edges.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_filled, mask, (0, 0), 255)
    im_filled_inv = cv2.bitwise_not(im_filled)
    im_out = edges | im_filled_inv
    im_out = cv2.dilate(im_out, kernel, iterations=3)
    """
    plt.imshow(edges)
    plt.show()
    plt.imshow(im_filled)
    plt.show()
    plt.imshow(im_filled_inv)
    plt.show()
    plt.imshow(im_out)
    plt.show()
    """
    # print(cv2.countNonZero(im_out))
    return im_out


def get_segmented_image(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = get_edges(gray)
    cv2.imshow('edge', edges)
    place = get_shape_from_edges(edges)
    segmented_place = np.bitwise_and(img, place[:, :, np.newaxis])
    segmented_edges = np.bitwise_or(img, edges[:, :, np.newaxis])
    return segmented_edges, segmented_place


def segment_image(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = get_edges(gray)
    place = get_shape_from_edges(edges)
    return edges, place


def main():
    test_names = ["images/test/3096.jpg", "images/test/8023.jpg", "images/test/12084.jpg", "images/test/14037.jpg",
                  "images/test/16077.jpg"]
    for e in test_names:
        img1, img2 = get_segmented_image(e)
        cv2.imshow('1', img1)
        cv2.imshow('2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
