"""
Utility functions for combining keras layers
"""

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import Add


def conv_block(inp, filters, kernel_size, strides=(2, 2),
               padding='same', use_act=True):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(inp)
    x = BatchNormalization()(x)
    return Activation(activation='relu')(x) if use_act else x

def res_block(inp, filters=64):
    kernel_size = (3, 3)
    strides = (1, 1)
    x = conv_block(inp, filters, kernel_size, strides)
    x = conv_block(x, filters, kernel_size, strides, use_act=False)
    x = Add()([x, inp])
    return x

def deconv_block(inp, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTransposed(filters, kernel_size, strides, padding=padding)(inp)
    x = BatchNormalization()(x)
    return Activation(activation='relu')(x)

def upsample_block(inp, filters, kernel_size, size=(2, 2)):
    x = UpSampling2D(size)(inp)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    return Activation(activation='relu')(x)
