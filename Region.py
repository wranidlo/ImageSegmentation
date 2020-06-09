import cv2
import numpy as np
import SimpleITK as sitk



def auto_region_growing(img):
    clicks=[]
    image=img.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



    imgInput = sitk.GetImageFromArray(image)

    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2, iterations=3)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            clicks.append((cY, cX))

    c = clicks[0]
    e = image[c[0], c[1]]
    seg = sitk.ConnectedThreshold(imgInput, seedList=clicks, lower=int(e - 50), upper=int(e + 50))

    img_arr = sitk.GetArrayFromImage(seg)
    outimg = np.array(255 * img_arr, dtype='uint8')
    edges = cv2.Canny(outimg, 100, 200)
    return outimg,edges

if __name__ == '__main__':
    image = cv2.imread('bird.jpg', 1)
    outimg,edges=auto_region_growing(image)
    cv2.imshow('Region Growing', outimg)
    cv2.waitKey()
    cv2.imshow('edges',edges)
    cv2.waitKey()
    cv2.destroyAllWindows()