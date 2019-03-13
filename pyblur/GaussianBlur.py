import numpy as np
from PIL import ImageFilter

from .common import PIL2array1C
gaussianbandwidths = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

def GaussianBlur_random(img, three_channel=True):
    gaussianidx = np.random.randint(0, len(gaussianbandwidths))
    gaussianbandwidth = gaussianbandwidths[gaussianidx]
    if not three_channel:
        return GaussianBlur(img, gaussianbandwidth)
    else:
        blurred_img = img
        for i in range(3):
            blurred_img[:,:,i] = PIL2array1C(GaussianBlur(img[:,:,i], gaussianbandwidths))
        blurred_img = Image.fromarray(blurred_img, 'RGB')
        return blurred_img


def GaussianBlur(img, bandwidth):
    img = img.filter(ImageFilter.GaussianBlur(bandwidth))
    return img
