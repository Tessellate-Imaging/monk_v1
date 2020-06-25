from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import warnings
import torch
import GPUtil
import networkx as nx
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

import os
import ipywidgets as widgets
from ipywidgets import interact, fixed
from IPython.display import display, clear_output
import matplotlib.gridspec as gridspec
from torchvision import transforms

layer_names = ["convolution1d", "convolution2d", "convolution", "convolution3d", "transposed_convolution1d",
                    "transposed_convolution", "transposed_convolution2d", "transposed_convolution3d",
                    "max_pooling1d", "max_pooling2d", "max_pooling", "max_pooling3d", "average_pooling1d",
                    "average_pooling2d", "average_pooling", "average_pooling3d", "global_max_pooling1d",
                    "global_max_pooling2d", "global_max_pooling", "global_max_pooling3d", "global_average_pooling1d",
                    "global_average_pooling2d", "global_average_pooling", "global_average_pooling3d", "fully_connected", 
                    "flatten", "dropout", "identity", "batch_normalization", "instance_normalization", "layer_normalization",
                    "relu", "sigmoid", "tanh", "softplus", "softsign",  "elu", "leaky_relu", "prelu", "selu",
                    "hardshrink", "hardtanh", "logsigmoid", "relu6", "rrelu", "celu", "softshrink", "tanhshrink",
                    "threshold", "softmin", "softmax", "logsoftmax", "add", "concatenate"]


names = ["conv1d_", "conv2d_", "conv_", "conv3d_", "tconv1d_", "tconv_", "tconv2d_", "tconv3d_",
			"max-pool1d_", "max-pool2d_", "max-pool_", "max-pool3d_", "avg-pooling1d_",
            "avg-pool2d_", "avg-pool_", "avg-pool3d_", "global-max-pool1d_", "global-max-pool2d_", 
            "global-max-pool_", "global-max-pool3d_", "global-avg-pool1d_", "global-avg-pool2d_", "global-avg-pool_", 
            "global-avg-pool3d_", "fc_", "flatten_", "dropout_", "identity_", "bn_", "inorm_",
            "lnorm_", "relu_", "sigmoid_", "tanh_", "softplus_", "softsign_", "elu_", "leaky_relu_", 
            "prelu_", "selu_", "hardshrink_", "hardtanh_", "logsigmoid_", "relu6_", "rrelu_", "celu_", 
            "softshrink_", "tanhshrink_", "threshold_", "softmin_", "softmax_", "logsoftmax_",
            "add_", "concatenate_"];