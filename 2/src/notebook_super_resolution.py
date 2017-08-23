"""
Lesson 1
"""

#===============================================================================
#===============================================================================
# INITIALIZE
#===============================================================================
#===============================================================================

# %% Set working directory
%cd src
################################################################################
# %% Imports
################################################################################
from importlib import reload

import utils; reload(utils)
import utils_layer; reload(utils_layer)
import utils_model; reload(utils_model)
import vgg16_avgpool; reload(vgg16_avgpool)

from utils_model import *
from utils_layer import *
from utils import *

import os
from os.path import join as jp
from glob import glob


import bcolz
import numpy as np
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

from PIL import Image
from IPython.display import HTML
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from scipy.ndimage import filters

from keras import backend as K
from keras import metrics
from keras.models import Model, Input
from keras.layers import Lambda
from keras.applications.vgg16 import VGG16



%matplotlib inline
plt.rcParams['figure.figsize'] = (3, 3)

# %% Limit GPU memory usage
################################################################################
utils.limit_mem()

# %% Set paths
################################################################################
path = '/home/riley/Work/fast_ai/2/data/imgnet_smpl/'

# %% Load data
################################################################################
n_imgs = 1000
arr_lr = bcolz.open(path+'trn_resized_72.bc')[:n_imgs]
arr_hr = bcolz.open(path+'trn_resized_288.bc')[:n_imgs]

# %% Define super-res model
################################################################################
inp = Input(arr_lr.shape[1:])
x = conv_block(inp, filters=64, kernel_size=(9, 9), strides=(1, 1))
x = res_block(x, filters=64)
x = res_block(x, filters=64)
# x = res_block(x, filters=64)
x = upsample_block(x, 64, kernel_size=(3, 3), size=(2, 2))
x = upsample_block(x, 64, kernel_size=(3, 3), size=(2, 2))
x = Conv2D(filters=3, kernel_size=(9, 9), activation='tanh', padding='same')(x)
out = Lambda(lambda x: (1 + x) * (127.5))(x)
out.shape

# %% Define content loss models
################################################################################
vgg_inp = Input(arr_hr.shape[1:])
vgg = vgg16_avgpool.VGG16_Avg(include_top=False, input_tensor=Lambda(vgg_preproc)(vgg_inp))

# %% Ensue vgg loss weights are static
for layer in vgg.layers: layer.trainable = False

# %% Create content loss layers
content_layers = [vgg.get_layer(f'block{i}_conv1').output
                  for i in range(1, 4)]
vgg_content = Model(vgg_inp, content_layers)

vgg_hr = vgg_content(vgg_inp)
vgg_lr = vgg_content(out)

# %% Loss function
w = [0.1, 0.8, 0.1]
def mse_content(diff):
    # Get mean on all channels
    dims = list(range(1, K.ndim(diff)))
    return K.expand_dims(K.sqrt(K.mean(diff**2, dims)), 0)

def fn_content_loss(x):
    n = len(w); resid = 0
    for i in range(n): resid += mse_content(x[i] - x[i+n]) * w[i]
    return resid

# %%  Model to fit
################################################################################

m_fit = Model([inp, vgg_inp], Lambda(fn_content_loss)(vgg_hr + vgg_lr))
target = np.zeros((arr_hr.shape[0], 1))
target.shape
m_fit.output.shape

# %% Fit
fit_params = {'verbose': 1}
m_fit.compile('adam', 'mean_squared_error')
m_fit.fit([arr_lr, arr_hr], target , batch_size=4, epochs=2, **fit_params)


# %%
pass
# %%
# %%
# %%
# %%
# %%



################################################################################
################################################################################
################################################################################
################################################################################


pass
