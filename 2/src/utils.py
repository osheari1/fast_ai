"""
Utility functions for fast ai mooc
"""

import os

from glob import glob

from PIL import Image

import numpy as np

import keras.backend as K

def limit_mem():
    """ Limits memory use on GPUs """
    K.get_session().close()
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=config))

def img_open(path, scale_factor=2):
    """ Opens and scales an image """
    with Image.open(path) as img:
        size = img.size
        img = img.resize(np.divide(size, scale_factor).astype(np.int32))
    return img
