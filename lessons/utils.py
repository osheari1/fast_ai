import numpy as np
import pandas as pd

import os
import shutil

from glob import glob


def struct_dir(path, classes, size_val=1000, size_smpl=50, ext='jpg'):
    """ Creates a better directory structure from Kaggle training data. It is assumed
        the images are named in the following structure 'class.####.ext'
    Args:
        path - Path to top level data directory where train/ and test/ sit
        classes - A list of class names.
        size_val - An integer or float deciding the size of the valid set. An integer
                   indicates a number of samples. A float indicates a percentage of the 
                   training set.
        size_smpl - An integer indicating how many images to copy to the sample set.
        ext - The file type.
    """
    if not os.path.exists(os.path.join(path, 'train')):
        print('Train data is missing.')
        return
    if os.path.exists(os.path.join(path, 'sample')) or \
            os.path.exists(os.path.join(path, 'valid')): 
        print("Directory structure already exists")
        return
    for cl in classes:
        # Create missing directories
        if not os.path.exists(os.path.join(path, 'sample', cl)):
            os.makedirs(os.path.join(path, 'sample', cl))
        if not os.path.exists(os.path.join(path, 'train', cl)):
            os.makedirs(os.path.join(path, 'train', cl))
        if not os.path.exists(os.path.join(path, 'valid', cl)):
            os.makedirs(os.path.join(path, 'valid', cl))
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
        # Copy a small number of training samples to sample 
        for file in np.random.choice(glob(os.path.join(path, 'train', cl, '*.'+ext)),
                                     size_smpl, replace=False): 
            shutil.copyfile(file, os.path.join(path, 'sample', cl, file.split('/')[-1]))
