import argparse
import cv2


def laplacian_variance(img_name):
    return cv2.Laplacian(img_name, cv2.CV_64F).var()


def test_if_not_blurred(img_name):

    image = cv2.imread(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = laplacian_variance(gray)
    if fm < 500.0:
        return False
    return True


def main():
    img_name = "D:\\Users\\baryl\Documents\GitHub\ImageSegmentation\images\\test\\3096.jpg"
    print(test_if_not_blurred(img_name))
    cv2.imshow('tested_img', cv2.imread(img_name))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

