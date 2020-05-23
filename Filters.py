import cv2
import numpy as np
import scipy.stats as st
import math
'''funkcje z dużej litery to właściwe filtry, reszta to funkcje pomocnicze'''
'''każdy filtr zwraca odszumioną kopię obrazu podanego jako argument'''




'''median filter'''
'''argumentu: obraz i rozmiar filtru (najlepiej 3 lub 5)'''
def Median_filter(image, filter_size):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = []
    indexer = filter_size // 2
    data=image.copy()
    for i in range(len(image)):
        for j in range(len(image[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(image) - 1:
                    for c in range(filter_size):
                        kernel.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(image[0]) - 1:
                        kernel.append(0)
                    else:
                        for k in range(filter_size):
                            kernel.append(image[i + z - indexer][j + k - indexer])
            kernel.sort()
            data[i][j] = kernel[len(kernel) // 2]
            kernel = []
    return data





'''gaussian filter'''
def gkern(kernlen, nsig):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()
def convolution(oldimage, kernel):
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    if (len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2,kernel_w // 2)), mode = 'constant', constant_values = 0).astype(np.float32)
    h = kernel_h // 2
    w = kernel_w // 2
    image_conv = np.zeros(image_pad.shape)
    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w
    if (h == 0):
        return image_conv[h:, w:w_end]
    if (w == 0):
        return image_conv[h:h_end, w:]
    return image_conv[h:h_end, w:w_end]
'''argumentu: obraz i rozmiar filtru (najlepiej 3 lub 5)'''
def GaussianBlurImage(image, filter_size):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = 1
    image = np.asarray(image)
    gaussian_filter=gkern(filter_size,sigma)
    im_filtered = convolution(image, gaussian_filter)
    return (im_filtered.astype(np.uint8))




'''bilateral filter'''
def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)
def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter//2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(int(source[neighbour_x][neighbour_y]) - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))
'''UWAGA - POWOLNY'''
'''argumenty: obraz, wartości sigm np: 30 i 30, a rozmiar filtru jest ustawiony na 3 bo przy większych rozmiarach filtrował przez pół wieczności'''
def Bilateral_filter(source, sigma_i, sigma_s):
    filter_diameter = 3
    if len(source.shape) == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    filtered_image = np.zeros(source.shape)
    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    target= np.asarray(filtered_image, dtype=np.uint8)
    return target

