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


if(tf.__version__.split(".")[0] == "2"):
	import tensorflow.compat.v1.keras.backend as K
else:
	from keras import backend as K

import keras.activations as kra
import keras.layers as krl