import cv2
import numpy as np
import concurrent.futures
import time


def get8n(x, y, shape,processed):
    out = []
    maxx = shape[0]-1
    maxy = shape[1]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    if not (outx,outy) in processed:
        out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))
    print(out)
    return out

clicks = []
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), image[y,x])
        clicks.append((y,x))
    if event == cv2.EVENT_RBUTTONDOWN:
        print(' Removed Last Seed')
        clicks.pop()

def region_growing(img, seed, outimg,processed):
    lista = []
    lista.append((seed[0],seed[1]))
    color = list(np.random.choice(range(256), size=3))
    while(len(lista) > 0):
        pix = lista[0]
        outimg[pix[0], pix[1]] = color
        reg=int(img[seed[0], seed[1]])
        size=1
        for coord in get8n(pix[0], pix[1], img.shape,processed):
            if (-25 < (int(img[coord[0], coord[1]]) - reg) < 25): #srednia z regionu
                outimg[coord[0], coord[1]] = color
                reg=(reg*size+int(img[coord[0], coord[1]]))/(size+1)
                size = size + 1
                if not coord in processed:
                    lista.append(coord)
                processed.append(coord)
        lista.pop(0)
        cv2.imshow("progress",outimg)
        cv2.waitKey(1)
    return outimg
def manual_region_growing(image):

    processed = []
    x = image.shape[0]
    y = image.shape[1]
    if x>250 or y>250:
        scale_percent=(250/image.shape[0])*100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)
        outimg = np.zeros_like(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image)

    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', image)
    cv2.waitKey(0)


    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results=[executor.submit(region_growing,image,seed,outimg,processed) for seed in clicks]

        for f in concurrent.futures.as_completed(results):
            outimg = outimg+f.result()

    end_time = time.time()
    time_taken = end_time - start_time
    print('time taken to complete: ',time_taken)
    edges = cv2.Canny(outimg, 100, 200)

    return outimg,edges
def auto_region_growing(image):

    processed = []
    x = image.shape[0]
    y = image.shape[1]
    if x>250 or y>250:
        scale_percent=(250/image.shape[0])*100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)
        outimg = np.zeros_like(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.array(image)


    cv2.imshow('Input', image)
    cv2.waitKey(0)
    #thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 1)
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('edges',thresh)
    cv2.waitKey(0)
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2, iterations=3)





    cv2.imshow('edges',thresh)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            clicks.append((cY, cX))

    #     cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
    #     cv2.putText(image, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 128, 200), 2)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results=[executor.submit(region_growing,image,seed,outimg,processed) for seed in clicks]

        for f in concurrent.futures.as_completed(results):
            outimg = outimg+f.result()

    end_time = time.time()
    time_taken = end_time - start_time
    print('time taken to complete: ',time_taken)
    edges = cv2.Canny(outimg, 100, 200)

    return outimg,edges


if __name__ == '__main__':


    image = cv2.imread('nemo.jpg', 1)
    outimg,edges=auto_region_growing(image)
    cv2.imshow('Region Growing', outimg)
    cv2.waitKey()
    cv2.imshow('edges',edges)
    cv2.waitKey()
    cv2.destroyAllWindows()

