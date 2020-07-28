import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import psutil
import numpy as np
import GPUtil

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
from tabulate import tabulate
from scipy.stats import logistic

################################################################################3
from monk.system.common import read_json
from monk.system.common import write_json
from monk.system.common import parse_csv
from monk.system.common import parse_csv_updated
from monk.system.common import save

from monk.system.summary import print_summary
################################################################################


################################################################################

from monk.pytorch.datasets.params import set_input_size
from monk.pytorch.datasets.params import set_batch_size
from monk.pytorch.datasets.params import set_data_shuffle
from monk.pytorch.datasets.params import set_num_processors
from monk.pytorch.datasets.params import set_weighted_sampling

from monk.pytorch.datasets.csv_dataset import DatasetCustom
from monk.pytorch.datasets.csv_dataset import DatasetCustomMultiLabel
from monk.pytorch.datasets.paths import set_dataset_train_path
from monk.pytorch.datasets.paths import set_dataset_test_path
################################################################################



################################################################################
from monk.pytorch.transforms.transforms import transform_center_crop
from monk.pytorch.transforms.transforms import transform_color_jitter
from monk.pytorch.transforms.transforms import transform_random_affine
from monk.pytorch.transforms.transforms import transform_random_crop
from monk.pytorch.transforms.transforms import transform_random_horizontal_flip
from monk.pytorch.transforms.transforms import transform_random_perspective
from monk.pytorch.transforms.transforms import transform_random_resized_crop
from monk.pytorch.transforms.transforms import transform_grayscale
from monk.pytorch.transforms.transforms import transform_random_rotation
from monk.pytorch.transforms.transforms import transform_random_vertical_flip
from monk.pytorch.transforms.transforms import transform_resize
from monk.pytorch.transforms.transforms import transform_normalize


from monk.pytorch.transforms.return_transform import set_transform_trainval
from monk.pytorch.transforms.return_transform import set_transform_test

from monk.pytorch.transforms.retrieve_transform import retrieve_trainval_transforms
from monk.pytorch.transforms.retrieve_transform import retrieve_test_transforms
################################################################################




################################################################################
from monk.system.eda.eda import class_imbalance
from monk.system.eda.eda import corrupted_missing_images
################################################################################



################################################################################
from monk.system.graphs.bar import create_plot
from monk.system.graphs.line import create_train_test_plots_accuracy
from monk.system.graphs.line import create_train_test_plots_loss
################################################################################




################################################################################
from monk.pytorch.models.layers import layer_dropout
from monk.pytorch.models.layers import layer_linear
from monk.pytorch.models.layers import activation_elu
from monk.pytorch.models.layers import activation_hardshrink
from monk.pytorch.models.layers import activation_hardtanh
from monk.pytorch.models.layers import activation_leakyrelu
from monk.pytorch.models.layers import activation_logsigmoid
from monk.pytorch.models.layers import activation_prelu
from monk.pytorch.models.layers import activation_relu
from monk.pytorch.models.layers import activation_relu6
from monk.pytorch.models.layers import activation_rrelu
from monk.pytorch.models.layers import activation_selu
from monk.pytorch.models.layers import activation_celu
from monk.pytorch.models.layers import activation_sigmoid
from monk.pytorch.models.layers import activation_softplus
from monk.pytorch.models.layers import activation_softshrink
from monk.pytorch.models.layers import activation_softsign
from monk.pytorch.models.layers import activation_tanh
from monk.pytorch.models.layers import activation_tanhshrink
from monk.pytorch.models.layers import activation_threshold
from monk.pytorch.models.layers import activation_softmin
from monk.pytorch.models.layers import activation_softmax
from monk.pytorch.models.layers import activation_logsoftmax


from monk.pytorch.models.params import set_model_name
from monk.pytorch.models.params import set_device
from monk.pytorch.models.params import set_pretrained
from monk.pytorch.models.params import set_freeze_base_network
from monk.pytorch.models.params import set_model_path


from monk.pytorch.models.common import set_parameter_requires_grad
from monk.pytorch.models.common import model_to_device
from monk.pytorch.models.common import print_grad_stats
from monk.pytorch.models.common import get_num_layers
from monk.pytorch.models.common import freeze_layers

from monk.pytorch.models.return_model import load_model
from monk.pytorch.models.return_model import setup_model
from monk.pytorch.models.return_model import debug_create_network

from monk.pytorch.models.features import CNNVisualizer
################################################################################






################################################################################
from monk.pytorch.schedulers.schedulers import scheduler_fixed
from monk.pytorch.schedulers.schedulers import scheduler_step
from monk.pytorch.schedulers.schedulers import scheduler_multistep
from monk.pytorch.schedulers.schedulers import scheduler_exponential
from monk.pytorch.schedulers.schedulers import scheduler_plateau

from monk.pytorch.schedulers.retrieve_scheduler import retrieve_scheduler

from monk.pytorch.schedulers.return_scheduler import load_scheduler
################################################################################








################################################################################
from monk.pytorch.optimizers.optimizers import adadelta
from monk.pytorch.optimizers.optimizers import adagrad
from monk.pytorch.optimizers.optimizers import adam
from monk.pytorch.optimizers.optimizers import adamw
from monk.pytorch.optimizers.optimizers import adamax
from monk.pytorch.optimizers.optimizers import rmsprop
from monk.pytorch.optimizers.optimizers import momentum_rmsprop
from monk.pytorch.optimizers.optimizers import sgd
from monk.pytorch.optimizers.optimizers import nesterov_sgd

from monk.pytorch.optimizers.retrieve_optimizer import retrieve_optimizer

from monk.pytorch.optimizers.return_optimizer import load_optimizer
################################################################################






################################################################################
from monk.pytorch.losses.losses import l1
from monk.pytorch.losses.losses import l2
from monk.pytorch.losses.losses import softmax_crossentropy
from monk.pytorch.losses.losses import crossentropy
from monk.pytorch.losses.losses import sigmoid_binary_crossentropy
from monk.pytorch.losses.losses import binary_crossentropy
from monk.pytorch.losses.losses import kldiv
from monk.pytorch.losses.losses import poisson_nll
from monk.pytorch.losses.losses import huber
from monk.pytorch.losses.losses import hinge
from monk.pytorch.losses.losses import squared_hinge
from monk.pytorch.losses.losses import multimargin
from monk.pytorch.losses.losses import squared_multimargin
from monk.pytorch.losses.losses import multilabelmargin
from monk.pytorch.losses.losses import multilabelsoftmargin

from monk.pytorch.losses.return_loss import load_loss

from monk.pytorch.losses.retrieve_loss import retrieve_loss
################################################################################






################################################################################
from monk.pytorch.training.params import set_num_epochs
from monk.pytorch.training.params import set_display_progress_realtime
from monk.pytorch.training.params import set_display_progress
from monk.pytorch.training.params import set_save_intermediate_models
from monk.pytorch.training.params import set_save_training_logs
from monk.pytorch.training.params import set_intermediate_model_prefix
################################################################################



################################################################################
from monk.pytorch.testing.process import process_single
from monk.pytorch.testing.process import process_multi
################################################################################