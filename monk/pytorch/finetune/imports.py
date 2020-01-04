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

################################################################################3
from system.common import read_json
from system.common import write_json
from system.common import parse_csv
from system.common import save

from system.summary import print_summary
################################################################################


################################################################################

from pytorch.datasets.params import set_input_size
from pytorch.datasets.params import set_batch_size
from pytorch.datasets.params import set_data_shuffle
from pytorch.datasets.params import set_num_processors
from pytorch.datasets.params import set_weighted_sampling

from pytorch.datasets.csv_dataset import DatasetCustom
from pytorch.datasets.paths import set_dataset_train_path
from pytorch.datasets.paths import set_dataset_test_path
################################################################################



################################################################################
from pytorch.transforms.transforms import transform_center_crop
from pytorch.transforms.transforms import transform_color_jitter
from pytorch.transforms.transforms import transform_random_affine
from pytorch.transforms.transforms import transform_random_crop
from pytorch.transforms.transforms import transform_random_horizontal_flip
from pytorch.transforms.transforms import transform_random_perspective
from pytorch.transforms.transforms import transform_random_resized_crop
from pytorch.transforms.transforms import transform_grayscale
from pytorch.transforms.transforms import transform_random_rotation
from pytorch.transforms.transforms import transform_random_vertical_flip
from pytorch.transforms.transforms import transform_resize
from pytorch.transforms.transforms import transform_normalize


from pytorch.transforms.return_transform import set_transform_trainval
from pytorch.transforms.return_transform import set_transform_test

from pytorch.transforms.retrieve_transform import retrieve_trainval_transforms
from pytorch.transforms.retrieve_transform import retrieve_test_transforms
################################################################################




################################################################################
from system.eda.eda import class_imbalance
from system.eda.eda import corrupted_missing_images
################################################################################



################################################################################
from system.graphs.bar import create_plot
from system.graphs.line import create_train_test_plots_accuracy
from system.graphs.line import create_train_test_plots_loss
################################################################################




################################################################################
from pytorch.models.layers import layer_dropout
from pytorch.models.layers import layer_linear
from pytorch.models.layers import activation_elu
from pytorch.models.layers import activation_hardshrink
from pytorch.models.layers import activation_hardtanh
from pytorch.models.layers import activation_leakyrelu
from pytorch.models.layers import activation_logsigmoid
from pytorch.models.layers import activation_prelu
from pytorch.models.layers import activation_relu
from pytorch.models.layers import activation_relu6
from pytorch.models.layers import activation_rrelu
from pytorch.models.layers import activation_selu
from pytorch.models.layers import activation_celu
from pytorch.models.layers import activation_sigmoid
from pytorch.models.layers import activation_softplus
from pytorch.models.layers import activation_softshrink
from pytorch.models.layers import activation_softsign
from pytorch.models.layers import activation_tanh
from pytorch.models.layers import activation_tanhshrink
from pytorch.models.layers import activation_threshold
from pytorch.models.layers import activation_softmin
from pytorch.models.layers import activation_softmax
from pytorch.models.layers import activation_logsoftmax


from pytorch.models.params import set_model_name
from pytorch.models.params import set_device
from pytorch.models.params import set_pretrained
from pytorch.models.params import set_freeze_base_network
from pytorch.models.params import set_model_path


from pytorch.models.common import set_parameter_requires_grad
from pytorch.models.common import model_to_device
from pytorch.models.common import print_grad_stats
from pytorch.models.common import get_num_layers
from pytorch.models.common import freeze_layers

from pytorch.models.return_model import load_model
from pytorch.models.return_model import setup_model
################################################################################






################################################################################
from pytorch.schedulers.schedulers import scheduler_fixed
from pytorch.schedulers.schedulers import scheduler_step
from pytorch.schedulers.schedulers import scheduler_multistep
from pytorch.schedulers.schedulers import scheduler_exponential
from pytorch.schedulers.schedulers import scheduler_plateau

from pytorch.schedulers.retrieve_scheduler import retrieve_scheduler

from pytorch.schedulers.return_scheduler import load_scheduler
################################################################################








################################################################################
from pytorch.optimizers.optimizers import adadelta
from pytorch.optimizers.optimizers import adagrad
from pytorch.optimizers.optimizers import adam
from pytorch.optimizers.optimizers import adamw
from pytorch.optimizers.optimizers import adamax
from pytorch.optimizers.optimizers import rmsprop
from pytorch.optimizers.optimizers import momentum_rmsprop
from pytorch.optimizers.optimizers import sgd
from pytorch.optimizers.optimizers import nesterov_sgd

from pytorch.optimizers.retrieve_optimizer import retrieve_optimizer

from pytorch.optimizers.return_optimizer import load_optimizer
################################################################################






################################################################################
from pytorch.losses.losses import softmax_crossentropy
from pytorch.losses.losses import nll
from pytorch.losses.losses import poisson_nll
from pytorch.losses.losses import binary_crossentropy
from pytorch.losses.losses import binary_crossentropy_with_logits

from pytorch.losses.return_loss import load_loss

from pytorch.losses.retrieve_loss import retrieve_loss
################################################################################






################################################################################
from pytorch.training.params import set_num_epochs
from pytorch.training.params import set_display_progress_realtime
from pytorch.training.params import set_display_progress
from pytorch.training.params import set_save_intermediate_models
from pytorch.training.params import set_save_training_logs
from pytorch.training.params import set_intermediate_model_prefix
################################################################################



################################################################################
from pytorch.testing.process import process_single
################################################################################