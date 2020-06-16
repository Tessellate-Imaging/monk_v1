from __future__ import print_function
from __future__ import division

import warnings
import mxnet as mx
import GPUtil
from mxnet import gluon, init, nd, initializer
from mxnet.gluon import nn
from gluoncv.model_zoo import get_model
import networkx as nx
from matplotlib import pyplot as plt

from mxnet.gluon.contrib import nn as contrib_nn

import os
import ipywidgets as widgets
from ipywidgets import interact, fixed
from IPython.display import display, clear_output
import matplotlib.gridspec as gridspec

ctx = [mx.cpu()];


layer_names = ["convolution1d", "convolution2d", "convolution", "convolution3d", "transposed_convolution1d",
                    "transposed_convolution", "transposed_convolution2d", "transposed_convolution3d", 
                    "max_pooling1d", "max_pooling2d", "max_pooling", "max_pooling3d", "average_pooling1d",
                    "average_pooling2d", "average_pooling", "average_pooling3d", "global_max_pooling1d",
                    "global_max_pooling2d", "global_max_pooling", "global_max_pooling3d", "global_average_pooling1d",
                    "global_average_pooling2d", "global_average_pooling", "global_average_pooling3d", 
                    "fully_connected", "dropout", "flatten", "identity", "add", "concatenate", "batch_normalization",
                    "instance_normalization", "layer_normalization", "relu", "sigmoid", "tanh", "softplus", "softsign", "elu", "gelu", "leaky_relu",
                    "prelu", "selu", "swish"]


names = ["conv1d_", "conv_", "conv_", "conv3d_", "tconv1d_", "tconv_", "tconv2d_", "tconv3d_", 
            "max-pool1d_", "mpool_", "mpool_", "max-pool3d_", "avg-pool1d_",
            "apool_", "apool_", "avg-pool3d_", "global-max-pool1d_", "global-max-pool2d_", "global-max-pool_", 
            "global-max-pool3d_", "global-avg-pool1d_", "global-avg-pool2d_", "global-avg-pool_", "global-avg-pool3d_", 
            "fc_", "dropout_", "flatten_", "identity_", "add_", "concat_", "bn_",
            "inorm_", "lnorm_", "relu_", "sigmoid_", "tanh_", "softplus_", "softsign_", "elu_", "gelu_", 
            "leaky_relu_", "prelu_", "selu_", "swish_"];





