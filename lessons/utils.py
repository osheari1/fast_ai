import numpy as np
import pandas as pd
import seaborn as sns

import os
import shutil
import PIL
import scipy

from scipy import ndimage
from PIL import Image
from glob import glob


from keras.preprocessing import image



def struct_dir(path, classes, size_val=1000, size_smpl=50, ext='jpg'):
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

      # Move a sample of test files
      for file in zip( 
          np.random.choice(glob(os.path.join(path,'test','test','*.'+ext)),
            size_smpl, replace=False)):
        shutil.copyfile(file, os.path.join(path,'sample','test','test',
                                           file.split('/')[-1]))

      # Move files to train / valid
      files = glob(os.path.join(path, 'train', cl+'.*.'+ext))
      if type(size_val) is float:
        size_val = np.floor(size_val * len(files))
      files_val = np.random.choice(files, size_val, replace=False)
      print('Moving %s: %d to train, %d to valid' % 
          (cl, len(files) - size_val, size_val))
      for file in files:
        if file in files_val:
          shutil.move(file, os.path.join(path, 'valid', cl))
          continue
        shutil.move(file, os.path.join(path, 'train', cl))
      # Copy a small number of training samples to sample train and valid
      for file_tr, file_val in zip( 
          np.random.choice(glob(os.path.join(path, 'train', cl, '*.'+ext)),
                           size_smpl, replace=False),
          np.random.choice(glob(os.path.join(path, 'valid', cl, '*.'+ext)),
                           size_smpl, replace=False)): 
        shutil.copyfile(file_tr, os.path.join(path, 'sample', 'train',
                                           cl, file_tr.split('/')[-1]))
        shutil.copyfile(file_tr, os.path.join(path, 'sample', 'valid',
                                           cl, file_tr.split('/')[-1]))

def plot_img(img):
  sns.plt.imshow(np.rollaxis(img, 0, 3).astype(np.uint8))

def plot_imgs(imgs, figsize=(12,6), rows=1, interp=False, titles=None):
  if type(imgs[0]) is np.ndarray:
    imgs = np.array(imgs).astype(np.uint8)
    if (imgs.shape[-1] != 3):
      imgs = imgs.transpose((0, 2, 3, 1))
  f = sns.plt.figure(figsize=figsize)
  for i in range(len(imgs)):
    sp = f.add_subplot(rows, len(imgs) // rows, i+1)
    if titles is not None:
      sp.set_title(titles[i], fontsize=18)
    sns.plt.imshow(imgs[i], interpolation=None if interp else 'none')

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True,
                batch_size=8, class_mode='categorical',
                target_size=(224, 224)):
  return gen.flow_from_directory(path, target_size=target_size,
          class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def get_imgs(path, target_size=(224, 224)):
  batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None,
                        target_size=target_size)
  return batches

def create_submit(batches, preds, fname='../submissions/submit.csv'):
  id = [x.split('/')[-1].split('.')[-2] for x in batches.filenames]
  df = pd.DataFrame({'id': id, 'label': preds})
  df.to_csv(fname, index=False, header=True)
  

