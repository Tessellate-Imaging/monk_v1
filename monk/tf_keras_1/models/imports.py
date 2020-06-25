from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import sys
import os
import GPUtil
import psutil


stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import tensorflow as tf
import networkx as nx
from matplotlib import pyplot as plt

if(tf.__version__.split(".")[0] == "2"):
	import tensorflow.compat.v1.keras.backend as K
else:
	from keras import backend as K

import keras.activations as kra
import keras.layers as krl

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import os
import ipywidgets as widgets
from ipywidgets import interact, fixed
from IPython.display import display, clear_output
import matplotlib.gridspec as gridspec
import numpy as np
from keras.models import Model

layer_names = ["convolution1d", "convolution2d", "convolution", "convolution3d", "transposed_convolution", 
				"transposed_convolution2d", "transposed_convolution3d", "max_pooling1d", "max_pooling2d", 
				"max_pooling", "max_pooling3d", "average_pooling1d", "average_pooling2d", "average_pooling", 
				"average_pooling3d", "global_max_pooling1d", "global_max_pooling2d", "global_max_pooling", 
				"global_max_pooling3d", "global_average_pooling1d", "global_average_pooling2d", "global_average_pooling", 
				"global_average_pooling3d", "flatten", "fully_connected", "dropout", "identity", "batch_normalization",
				"relu", "elu", "leaky_relu", "prelu", "thresholded_relu", "softmax", "add", "concatenate",
				"selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid"]


names = ["conv1d_", "conv2d_", "conv_", "conv3d_", "tconv_", 
		"tconv2d_", "tconv3d_", "max-pool1d_", "max-pool2d_", 
		"max-pool_", "max-pool3d_", "avg-pool1d_", "avg-pool2d_", "avg-pool_", 
		"avg-pool3d_", "global-max-pool1d_", "global-max-pool2d_", "global-max-pool_", 
		"global-max-pool3d_", "global-avg-pool1d_", "global-avg-pool2d_", "global-avg-pool_", 
		"global-avg-pool3d_", "flatten_", "fc_", "dropout_", "identity_", "bn_",
		"relu_", "elu_", "lrelu_", "prelu_", "trelu_", "softmax_", "add_", "concat_",
		"selu_", "softplus_", "softsign_", "tanh_", "sigmoid_", "hard_sigmoid_"]