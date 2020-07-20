import os
import sys

import numpy as np 
import torch
from PIL import Image
from torch.autograd import Variable
from scipy.stats import logistic
from scipy.special import softmax