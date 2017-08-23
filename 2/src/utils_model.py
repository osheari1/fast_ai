"""
Utility functions related to specific pretrained models
"""
import numpy as np
import PIL


def vgg_preproc(img):
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return (img - vgg_mean)[:, :, :, ::-1]

def vgg_deproc(img, shape):
    return np.clip(img.reshape(shape)[:, :, :, ::-1] + vgg_mean, 0, 255)
