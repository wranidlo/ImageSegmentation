import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_edges(img_name):
    img = cv2.imread('images/'+img_name, 0)
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    return edges


def get_shape_from_edges(edges):

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    plt.show()
    im_filled = edges.copy()
    h, w = edges.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_filled, mask, (0, 0), 255)
    im_filled_inv = cv2.bitwise_not(im_filled)
    im_out = edges | im_filled_inv

    plt.imshow(edges)
    plt.show()
    plt.imshow(im_filled)
    plt.show()
    plt.imshow(im_filled_inv)
    plt.show()
    plt.imshow(im_out)
    plt.show()

    return im_out


def main():
    edges = get_edges("plane.jpg")
    get_shape_from_edges(edges)


if __name__ == "__main__":
    main()