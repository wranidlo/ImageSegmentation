import argparse
import cv2


def laplacian_variance(img_name):
    return cv2.Laplacian(img_name, cv2.CV_64F).var()


list_img = ["Blurred-vision.jpeg", "sea.jpg", "sli.jpg"]

for imagePath in list_img:
    image = cv2.imread("images/" + imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = laplacian_variance(gray)
    text = "Not Blurry"

    if fm < 50.0:
        text = "Blurry"
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    image = cv2.resize(image, (800, 600))
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)
