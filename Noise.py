import numpy as np
from skimage.util import random_noise



def noise_SaltPepper(img,amount=0.01):
    noise_img = random_noise(img, mode='s&p', amount=amount)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img
def noise_Gaussian(img,mean=0,var=0.0008):
    noise_img = random_noise(img, mode='gaussian',mean=mean,var=var)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img
def noise_Speckle(img,mean=0,var=0.0008):
    noise_img = random_noise(img, mode='speckle',mean=mean,var=var)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img