# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import circle

from .common import PIL2array1C

defocusKernelDims = [3,5,7,9]

def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))    
    kerneldim = defocusKernelDims[kernelidx]
    if not three_channel:
        return DefocusBlur(img, kerneldim)
    else:
        blurred_img = img
        for i in range(3):
            blurred_img[:,:,i] = PIL2array1C(DefocusBlur(img[:,:,i], kerneldim))
        blurred_img = Image.fromarray(blurred_img, 'RGB')
        return blurred_img

def DefocusBlur(img, dim):
    imgarray = np.array(img, dtype="float32")
    kernel = DiskKernel(dim)
    convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = dim / 2
    circleRadius = circleCenterCoord +1
    
    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr,cc]=1
    
    if(dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)
        
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel

def Adjust(kernel, kernelwidth):
    kernel[0,0] = 0
    kernel[0,kernelwidth-1]=0
    kernel[kernelwidth-1,0]=0
    kernel[kernelwidth-1, kernelwidth-1] =0 
    return kernel
