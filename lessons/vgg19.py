import numpy as np
import tensorflow as tf

import os, json
import utils

from glob import glob
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils import data_utils
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

vgg_mean = np.array([123.68, 116.779, 103.939],
                    dtype=np.float32).reshape((1,1,3))

def vgg_preprocess(x):
  return x - vgg_mean


class Vgg19():
  """ The VGG 19 Imagenet model """

  def __init__(self):
    self.FILE_PATH = 'http://www.platform.ai/models/'
    self.WEIGHTS = '../data/pretrained_weights/vgg19.h5'
    self.create()
    self.get_classes()
  
  def get_classes(self):
    fname = 'imagenet_class_index.json'
    fpath = data_utils.get_file(fname, self.FILE_PATH+fname,
                                cache_subdir='models')
    with open(fpath) as f:
      class_dict = json.load(f)
    self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

  def ConvBlock(self, layers, filters):
    model = self.model
    for i in range(layers):
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  def FCBlock(self):
    model = self.model
    model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))


  def create(self):
    model = self.model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(224,224,3)))

    self.ConvBlock(2, 64)
    self.ConvBlock(2, 128)
    self.ConvBlock(4, 256)
    self.ConvBlock(4, 512)
    self.ConvBlock(4, 512)

    model.add(Flatten())
    self.FCBlock()
    self.FCBlock()
    model.add(Dense(1000, activation='softmax'))

    print("Loading model weights")
    model.load_weights(self.WEIGHTS)

  def predict(self, imgs, details=False):
    all_preds = self.model.predict(imgs)
    idxs = np.argmax(all_preds, axis=1)
    preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
    classes = [self.classes[idx] for idx in idxs]
    return np.array(preds), idxs, classes


  def ft(self, num, compile_kwargs={}):
    """ Retrain the last layer of a model with a new last layer of 
        arbitrary size.
    """
    model = self.model
    model.pop()
    for layer in model.layers:
      layer.trainable=False
    model.add(Dense(num, activation='softmax'))
    self.compile(**compile_kwargs)

  def compile(self, lr=0.001):
    self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',
                       metrics=['accuracy'])

  def finetune(self, batches, compile_kwargs={}):
    model = self.model
    model.pop()
    for layer in model.layers:
      layer.trainable=False
    model.add(Dense(batches.nb_class, activation='softmax'))
    self.compile(**compile_kwargs)

  def fit_data(self, train, labels, valid, labels_valid, nb_epoch=1,
               batch_size=64):
    self.model.fit(train, labels, nb_epoch=nb_epoch,
                   validation_data=(valid, labels_valid), batch_size=batch_size)

  def fit(self, batches, batches_valid, nb_epoch, callbacks=None):
    self.model.fit_generator(batches, samples_per_epoch=batches.N,
                             nb_epoch=nb_epoch,
                             validation_data=batches_valid,
                             nb_val_samples=batches_valid.N,
                             callbacks=callbacks)

  def test(self, path, batch_size=8):
    batches_test = utils.get_batches(path, shuffle=False, batch_size=batch_size,
                                    class_mode=None)
    return batches_test, self.model.predict_generator(batches_test,
                                                      batches_test.nb_sample)


