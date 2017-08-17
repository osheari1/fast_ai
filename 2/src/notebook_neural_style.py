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
import vgg16_avgpool; reload(vgg16_avgpool)

from utils import *

import os
from os.path import join as jp
from glob import glob


import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import animation
from IPython.display import HTML
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from scipy.ndimage import filters

from keras import metrics
from keras.models import Model
from keras import backend as K

%matplotlib inline
plt.rcParams['figure.figsize'] = (3, 3)

# %% Limit GPU memory usage
################################################################################
utils.limit_mem()
# %% Set paths
################################################################################
path = '/home/riley/Work/fast_ai/2/data/imgnet_smpl/'
fnames = glob(path+'train/**/*.JPEG', recursive=True)

#===============================================================================
#===============================================================================
# NEURAL STYLE TRANSFER
#===============================================================================
#===============================================================================
# %%
img = img_open(fnames[450], 2); img
img.size
# %% Define preprocessing and deprocessing functions
# Vgg mean values: 123.68, 116.779, 103.939
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - vgg_mean)[:, :, :, ::-1]
deproc = lambda x, s: np.clip(x.reshape(s)[:, :, :, ::-1] + vgg_mean, 0, 255)
plot_arr = lambda x: plt.imshow(deproc(x, x.shape)[0].astype(np.int8))
# %% Define preprocessing and deprocessing functions
img_arr = preproc(np.expand_dims(np.array(img), 0))
shape = img_arr.shape; shape

#===============================================================================
# Recreate input - contect loss
#===============================================================================
# %%
vgg = vgg16_avgpool.VGG16_Avg(include_top=False)

# %%
[l.name for l in vgg.layers]
layer = vgg.get_layer('block5_conv1').output

# %% Calculate target activations. ie The output given a single image
layer_model = Model(vgg.input, layer)
pred = K.variable(layer_model.predict(img_arr))

# %% Define class that handles loss and gradient outputs
class Evaluator(object):
    def __init__(self, fnc, shape): self.fnc, self.shape = fnc, shape
    def loss(self, img):
        loss_, self.grads_ = self.fnc([img.reshape(self.shape)])
        return loss_.astype(np.float64)
    def grads(self, img): return self.grads_.flatten().astype(np.float64)

# %% Define loss, gradients
loss = K.mean(metrics.mse(layer, pred))  # Actual layer is y_true
grads = K.gradients(loss, vgg.input)
fnc = K.function([vgg.input], [loss]+grads)
evaluator = Evaluator(fnc, shape)

# %% Perform deterministic optimization on img
def solve_img(eval_obj, n_iter, x, out_shape, path):
    for i in range(n_iter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss,
                                           x.flatten(),
                                           fprime=eval_obj.grads,
                                           maxfun=20)
        x = np.clip(x, -127, 127)

        print('Current loss value %s/%s:' % (i+1, n_iter), min_val)
        imsave(f'{path}at_iteration_{i}.png', deproc(x.copy(), out_shape)[0])
    return x

# %% Generate random imge
gen_rand_img = lambda shape, sclr: np.random.uniform(-2.5, 2.5, shape) / sclr
rand_img = gen_rand_img(shape, 100); plt.imshow(rand_img[0])

# %% Run context optimization
n_iter = 5
solve_img(evaluator, n_iter, rand_img, img_arr.shape, path+'results/context/')

# %%
img_open(path+'results/context/at_iteration_4.png', 2)

# %% Animate progression
def animate(i):  ax.imshow(Image.open(f'{path}results/context/at_iteration_{i}.png'))

# %%
fig, ax = plt.subplots();
anim = animation.FuncAnimation(fig, animate, frames=4, interval=200)
HTML(anim.to_html5_video())
# %%
#===============================================================================
# Recreate style - Style Loss
#===============================================================================
path_styles = '/home/riley/Work/fast_ai/2/data/styles'
# %%
style = img_open(path_styles+'/starry_night.png', 1.1); style
style.size
# %%
style = img_open(path_styles+'/bird.png', 1); style
style.size
# %%
style = img_open(path_styles+'/simpsons.png', 1.9); style
# %%
style_arr = preproc(np.expand_dims(style, 0)[:, :, :, :3])
style_shape = style_arr.shape
style_shape

# %% Single image input shape
vgg = vgg16_avgpool.VGG16_Avg(include_top=False, input_shape=style_shape[1:])
layers = {l.name: l.output for l in vgg.layers}
layers

# %% Create layers model and targets
style_layers = [layers[f'block{i}_conv1'] for i in [1, 2, 3]]
layers_model = Model(vgg.input, style_layers)
style_targets = [K.variable(p) for p in layers_model.predict(style_arr)]

# %% Gram matrix
def gram_matrix(x):
    # Each row should be a channel
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()

# %% Define variables
style_loss = lambda l, t: K.mean(metrics.mse(gram_matrix(l), gram_matrix(t)))
loss = sum([style_loss(l[0], t[0]) for l, t in zip(style_layers, style_targets)])

grads = K.gradients(loss, vgg.input)
style_fnc = K.function([vgg.input], [loss]+grads)
evaluator = Evaluator(style_fnc, style_arr.shape)
# %% Generate random image
rand_img = gen_rand_img(style_shape, 1)
rand_img = filters.gaussian_laplace(rand_img, (0, 2, 2, 0))
# plt.imshow(rand_img[0])

# %% Solve image
n_iter = 10
img = solve_img(evaluator, n_iter, rand_img, style_shape, f'{path}/results/style/')
Image.open(f'{path}results/style/at_iteration_9.png')

# %% Animate
def animate(i):  ax.imshow(Image.open(f'{path}results/style/at_iteration_{i}.png'))
fig, ax = plt.subplots();
anim = animation.FuncAnimation(fig, animate, frames=10, interval=200)
HTML(anim.to_html5_video())

# %%
#===============================================================================
# Recreate input - contect loss
#===============================================================================
# %% Crop image to style size
w, h = style.size
src = img_arr[:, :h, :w]
src.shape
style_arr.shape
img_arr.shape
# %% Set new model
vgg = vgg16_avgpool.VGG16_Avg(include_top=False, input_shape=src.shape[1:])

# %% Get content and style layers
layers = {l.name: l.output for l in vgg.layers}
style_layers = [layers[f'block{i}_conv2'] for i in range(1, 6)]
content_layer = layers['block4_conv2']

# %% Create models and targets
content_model = Model(vgg.input, content_layer)
content_target = K.variable(content_model.predict(src))

style_model = Model(vgg.input, style_layers)
style_targets = [K.variable(p) for p in style_model.predict(style_arr)]

# %% Create loss function
style_w = [0.1,0.2,0.2,0.25,0.25]  # Each style layer has a weight
content_sclr = 10
loss = sum(style_loss(l[0], t[0])*w
           for l, t, w in zip(style_layers, style_targets, style_w))
loss += K.mean(metrics.mse(content_layer, content_target)) / content_sclr
grads = K.gradients(loss, vgg.input)
transfer_fnc = K.function([vgg.input], [loss]+grads)
evaluator = Evaluator(transfer_fnc, src.shape)

# %% Generate random image
rand_img = gen_rand_img(src.shape, 1)
rand_img = filters.gaussian_filter(rand_img, (0, 2, 2, 0))
plt.imshow(rand_img[0])
# %% Solve
n_iter = 20
img = solve_img(evaluator, n_iter, rand_img,
                src.shape, f'{path}results/transfer/')
# %%
pass
# %%
