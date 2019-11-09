import os 
import sys
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
if(isnotebook()):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
