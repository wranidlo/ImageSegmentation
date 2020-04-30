import cv2
import numpy as np
import concurrent.futures
import time
from multiprocessing import Manager






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

def region_growing(img, seed, outimg,processed):
    list = []
    list.append((seed[0],seed[1]))
    reg=int(img[seed[0], seed[1]])
    size=1
    while(len(list) > 0):
        pix = list[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape,processed):
            if (-20 < (int(img[coord[0], coord[1]]) - reg) < 20): #srednia z regionu
                outimg[coord[0], coord[1]] = 255
                reg=(reg*size+int(img[coord[0], coord[1]]))/(size+1)
                size = size + 1
                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)
        # wyświetlanie rozrostu (spowalnia program)
        # cv2.imshow("progress",outimg)
        # cv2.waitKey(1)
    return outimg

#wybór punktów startowych (seeds) za pomocą myszki
def on_mouse(event, x, y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), image[y,x])
        clicks.append((y,x))
    if event == cv2.EVENT_RBUTTONDOWN:
        print(' Removed Last Seed')
        clicks.pop()

if __name__ == '__main__':

    clicks = []
    #manager = Manager()
    #processed = manager.list()
    processed=[]
    image = cv2.imread('rocks.jpg', 0)

    x = image.shape[0]
    y = image.shape[1]
    if x>250 or y>250:
        scale_percent=(250/image.shape[0])*100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)
        image = np.array(image)

    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', image)
    cv2.waitKey(0)

    outimg = np.zeros_like(image)
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results=[executor.submit(region_growing,image,seed,outimg,processed) for seed in clicks]

        for f in concurrent.futures.as_completed(results):
            outimg = outimg+f.result()

    end_time = time.time()
    time_taken = end_time - start_time
    print('time taken to complete: ',time_taken)
    #przeskalowanie do oryginalnego rozmiaru
    #outimg = cv2.resize(outimg, (y, x), interpolation=cv2.INTER_AREA)
    cv2.imshow('Region Growing', outimg)
    cv2.waitKey()
    edges = cv2.Canny(outimg, 100, 200)
    cv2.imshow('edges',edges)
    cv2.waitKey()
    cv2.imwrite('rocks.bmp',edges)
    cv2.destroyAllWindows()