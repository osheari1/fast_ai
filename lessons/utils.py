import numpy as np
import pandas as pd
import seaborn as sns

import bcolz
import operator
import sys
import os
import shutil
import PIL
import scipy
import h5py

from scipy import ndimage
from PIL import Image
from glob import glob
from IPython import display


from sklearn import metrics
from keras import models, layers
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import layer_from_config


def create_dirs(path, classes, ext='jpg'):
  if not os.path.exists(os.path.join(path, 'train')):
    print('Train data is missing.')
    return
  if os.path.exists(os.path.join(path, 'sample')) or\
      os.path.exists(os.path.join(path, 'valid')): 
    print("Directory structure already exists")
    return

  for cl in classes:
      # Create missing directories
      if not os.path.exists(os.path.join(path,'sample','train',cl)):
        os.makedirs(os.path.join(path,'sample','train',cl))
      if not os.path.exists(os.path.join(path,'sample','valid',cl)):
        os.makedirs(os.path.join(path,'sample','valid',cl))
      if not os.path.exists(os.path.join(path,'sample','test','test')):
        os.makedirs(os.path.join(path,'sample','test','test'))
      if not os.path.exists(os.path.join(path,'train',cl)):
        os.makedirs(os.path.join(path,'train',cl))
      if not os.path.exists(os.path.join(path,'valid',cl)):
        os.makedirs(os.path.join(path,'valid',cl))
      
      # If test is not in a dir called test, create and move all test files
      if not os.path.exists(os.path.join(path, 'test', 'test')):
        os.makedirs(os.path.join(path, 'test', 'test'))
        files = glob(os.path.join(path, 'test', '*'+ext))
        for f in files:
          shutil.move(f, os.path.join(path, 'test', 'test'))


def struct_dir(path, classes, size_val=1000, size_smpl=50, ext='jpg'):# {{{
  """ Creates a better directory structure from Kaggle training data. It is
      assumed the images are named in the following structure 'class.####.ext'
  Args:
      path - Path to top level data directory where train/ and test/ sit
      classes - A list of class names.
      size_val - An integer or float deciding the size of the valid set. 
                 An integer indicates a number of samples. A float indicates a
                 percentage of the training set.
      size_smpl - An integer indicating how many images to copy to the
                  sample set.
      ext - The file type.
  """

  if not os.path.exists(os.path.join(path, 'train')):
    print('Train data is missing.')
    return
  if os.path.exists(os.path.join(path, 'sample')) or\
      os.path.exists(os.path.join(path, 'valid')): 
    print("Directory structure already exists")
    return

  for cl in classes:
      # Create missing directories
      if not os.path.exists(os.path.join(path,'sample','train',cl)):
        os.makedirs(os.path.join(path,'sample','train',cl))
      if not os.path.exists(os.path.join(path,'sample','valid',cl)):
        os.makedirs(os.path.join(path,'sample','valid',cl))
      if not os.path.exists(os.path.join(path,'sample','test','test')):
        os.makedirs(os.path.join(path,'sample','test','test'))
      if not os.path.exists(os.path.join(path,'train',cl)):
        os.makedirs(os.path.join(path,'train',cl))
      if not os.path.exists(os.path.join(path,'valid',cl)):
        os.makedirs(os.path.join(path,'valid',cl))
      
      # If test is not in a dir called test, create and move all test files
      if not os.path.exists(os.path.join(path, 'test', 'test')):
        os.makedirs(os.path.join(path, 'test', 'test'))
        files = glob(os.path.join(path, 'test', '*'+ext))
        for f in files:
          shutil.move(f, os.path.join(path, 'test', 'test'))


      # Move a sample of test files
      for f in np.random.choice(glob(os.path.join(path,'test','test','*.'+ext)),
                                size_smpl, replace=False):
        shutil.copyfile(f, os.path.join(path,'sample','test','test',
                        f.split('/')[-1]))

      # Move files to train / valid
      files = glob(os.path.join(path, 'train', cl+'.*.'+ext))
      if type(size_val) is float:
        size_val = np.floor(size_val * len(files))
      files_val = np.random.choice(files, size_val, replace=False)
      print('Moving %s: %d to train, %d to valid' % 
          (cl, len(files) - size_val, size_val))
      for f in files:
        if f in files_val:
          shutil.move(f, os.path.join(path, 'valid', cl))
          continue
        shutil.move(f, os.path.join(path, 'train', cl))
      # Copy a small number of training samples to sample train and valid
      for f_tr, f_val in zip( 
          np.random.choice(glob(os.path.join(path, 'train', cl, '*.'+ext)),
                           size_smpl, replace=False),
          np.random.choice(glob(os.path.join(path, 'valid', cl, '*.'+ext)),
                           size_smpl, replace=False)): 
        shutil.copyfile(f_tr, os.path.join(path, 'sample', 'train',
                                           cl, f_tr.split('/')[-1]))
        shutil.copyfile(f_val, os.path.join(path, 'sample', 'valid',
                                           cl, f_val.split('/')[-1]))# }}}

def rm_checkpoints(path_match, epoch_keep):# {{{
  # Convert numberic input to string
  if isinstance(epoch_keep, int):
    epoch_keep = epoch_keep - 1  # Adjust for file name difference
    if epoch_keep < 10:
      epoch_keep = '0%d' % epoch_keep
    else:
      epoch_keep = str(epoch_keep)
  
  for f in glob(path_match+'*'):
    if not '.%s-'%epoch_keep in f:
      os.remove(f)# }}}

def plot_img(img):# {{{
  """ Plot PIL image
  """
  sns.plt.imshow(np.rollaxis(img, 0, 3).astype(np.uint8))# }}}

def plot_imgs(imgs, figsize=(12,6), rows=1, interp=False, titles=None):# {{{
  """ Plot a list of PIL images
  """
  if type(imgs[0]) is np.ndarray:
    imgs = np.array(imgs).astype(np.uint8)
    if (imgs.shape[-1] != 3):
      imgs = imgs.transpose((0, 2, 3, 1))
  f = sns.plt.figure(figsize=figsize)
  for i in range(len(imgs)):
    sp = f.add_subplot(rows, len(imgs) // rows, i+1)
    if titles is not None:
      sp.set_title(titles[i], fontsize=18)
    sp.grid(False)
    ax = sns.plt.imshow(imgs[i], interpolation=None if interp else 'none')
    # }}}

def plot_conf_matrix(ytrue, preds, labels):# {{{
  """ Plot confusion matrix
  """
  labels_sorted = sorted(labels.items(), key=operator.itemgetter(1))
  lname = [k for k, v in labels_sorted]
  lidx = [v for k, v in labels_sorted]
  cm = metrics.confusion_matrix(ytrue, preds, labels=lidx)
  df = pd.DataFrame(cm, columns=labels, index=labels)
  pl = sns.heatmap(df, annot=True)
  return pl# }}}

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True,# {{{
                batch_size=8, class_mode='categorical',
                target_size=(224, 224)):
  return gen.flow_from_directory(path, target_size=target_size,
          class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)# }}}

def get_data(path, target_size=(224, 224)):# {{{
  batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None,
                        target_size=target_size)
  
  return np.concatenate([batches.next() for i in range(batches.nb_sample)])# }}}

def create_submit(batches, probas, fname, clip=(0, 1)):# {{{
  probas = probas.clip(*clip)
  
  id = [x.split('/')[-1].split('.')[-2] for x in batches.filenames]
  df = pd.DataFrame({'id': id, 'label': probas})
  df.to_csv(fname, index=False, header=True)# }}}

def get_classes(path):# {{{
  batches_train = get_batches(path+'train', shuffle=False, batch_size=1)
  batches_valid = get_batches(path+'valid', shuffle=False, batch_size=1)
  batches_test = get_batches(path+'test', shuffle=False, batch_size=1)
  return (batches_train.classes, batches_valid.classes,
          to_categorical(batches_train.classes), 
          to_categorical(batches_valid.classes)), \
         (batches_train.filenames, batches_valid.filenames, 
          batches_test.filenames)# }}}

#### Array funcs ##### {{{
def compress_imgs(path, fname, chunk_size=None, target_size=(224,224)):# {{{
  batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None,
                        target_size=target_size)

  if not chunk_size:
    chunk_size = batches.nb_sample
  
  if chunk_size > batches.nb_sample:
    chunk_size = batches.nb_sample

  if not os.path.exists(fname):
    os.makedirs(fname)

  chunks_total = batches.nb_sample // chunk_size + 1 \
                    if batches.nb_sample % chunk_size != 0 \
                    else batches.nb_sample // chunk_size
  print('Saving %d chunks' % chunks_total)
  chunk_i = 1
  arr = []
  for i in range(batches.nb_sample):
    arr.append(batches.next())
    if (i+1) % chunk_size == 0 or i == batches.nb_sample-1:
      print('', end='\r')
      print('['+'='*chunk_i+'>'+' '*(chunks_total-chunk_i)+']'+' %d/%d images'\
            % (i+1 , batches.nb_sample),
            end='' if i != batches.nb_sample-1 else '\n')

      if i+1 == chunk_size:
        c = bcolz.carray(np.concatenate(arr), rootdir=fname, mode='w')
        c.flush()
      else:
        c.append(np.concatenate(arr))
        c.flush()
      arr = []
      chunk_i += 1# }}}

def save_array_bcolz(fname, arr, mode='w'):# {{{
  c=bcolz.carray(arr, rootdir=fname, mode=mode)
  c.flush()# }}}

def load_array_bcolz(fname):# {{{
  return bcolz.open(fname)[:]# }}}

def save_array_h5(fname, arr):# {{{
  with h5py.File(fname, 'w') as hf:
    hf.create_dataset(fname.split('/')[-1].split('.')[0],
                      data=arr)# }}}

def load_array_h5(fname):# {{{
  with h5py.File(fname, 'r') as hf:
    return hf[fname.split('/')[-1].split('.')[0]][:]# }}}# }}}

##### Layer funcs ###### {{{
def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]

def wrap_config(layer):
  return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}

def copy_layer(layer): 
  return layer_from_config(wrap_config(layer))

def copy_layers(layers, input_shape=None): 
  return [copy_layer(layer).__class__(input_shape=input_shape) \
            if i == 0 and input_shape \
            else copy_layer(layer) 
            for i, layer in enumerate(layers)]

def insert_layer(model, new_layer, index):
  m_new = models.Sequential()
  for i, layer in enumerate(model.layers):
    if i == index:
      m_new.add(new_layer)
    layer_cpy = copy_layer(layer)
    m_new.add(layer_cpy)
    layer_cpy.set_weights(layer.get_weights())
  return m_new

def insert_layers(model, new_layer_class, idxs):
  m_new = models.Sequential()
  for i, layer in enumerate(model.layers):
    if i in idxs:
      m_new.add(new_layer_class())
    layer_cpy = copy_layer(layer)
    m_new.add(layer_cpy)
    layer_cpy.set_weights(layer.get_weights())
  return m_new

def copy_weights(from_layers, to_layers):
  for from_layer,to_layer in zip(from_layers, to_layers):
    to_layer.set_weights(from_layer.get_weights())

def copy_model(m):
  layers_cpy = copy_layers(m.layers)
  m_copy = models.Sequential(layers_cpy)
  copy_weights(m.layers, m_copy.layers)
  return m_copy

def create_model_from_layers(layers):
  layers_cpy = copy_layers(layers, input_shape=layers[0].input_shape[1:])
  model = models.Sequential(layers_cpy)
  copy_weights(layers, model.layers)
  return model

def set_dropout(model, p_prev=0.5, p_new=0):
  """Assumes uniform dropout prob"""
  for layer in model.layers:
    if not type(layer) is layers.Dropout:
      layer.set_weights(adjust_dropout(layer.get_weights(), p_prev, p_new))
      continue
    layer.p = p_new

def adjust_dropout(weights, p_prev, p_new):
    scal = (1-p_prev)/(1-p_new)
    return [w*scal for w in weights]


# }}}


