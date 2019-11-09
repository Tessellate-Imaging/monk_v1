from __future__ import print_function
from __future__ import division

import warnings
import mxnet as mx
import GPUtil
from mxnet import gluon, init, nd
from mxnet.gluon import nn
from gluoncv.model_zoo import get_model

ctx = [mx.cpu()];
