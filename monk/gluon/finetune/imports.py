import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time


import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import psutil
import shutil
import numpy as np
import GPUtil
import cv2



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


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0";
import mxnet as mx
import torch
from mxnet import autograd as ag
from tabulate import tabulate
from scipy.stats import logistic
from mxnet import image
from mxnet.gluon.data.vision import transforms




################################################################################3
from system.common import read_json
from system.common import write_json
from system.common import parse_csv
from system.common import parse_csv_updated
from system.common import save

from system.summary import print_summary
################################################################################




################################################################################
from gluon.datasets.class_imbalance import balance_class_weights

from gluon.datasets.params import set_input_size
from gluon.datasets.params import set_batch_size
from gluon.datasets.params import set_data_shuffle
from gluon.datasets.params import set_num_processors
from gluon.datasets.params import set_weighted_sampling

from gluon.datasets.csv_dataset import DatasetCustom
from gluon.datasets.csv_dataset import DatasetCustomMultiLabel
from gluon.datasets.paths import set_dataset_train_path
from gluon.datasets.paths import set_dataset_test_path
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
from gluon.losses.losses import softmax_crossentropy
from gluon.losses.losses import crossentropy
from gluon.losses.losses import sigmoid_binary_crossentropy
from gluon.losses.losses import binary_crossentropy
from gluon.losses.losses import poisson_nll
from gluon.losses.losses import l1
from gluon.losses.losses import l2
from gluon.losses.losses import kldiv
from gluon.losses.losses import huber
from gluon.losses.losses import hinge
from gluon.losses.losses import squared_hinge


from gluon.losses.return_loss import load_loss

from gluon.losses.retrieve_loss import retrieve_loss
################################################################################




################################################################################
from gluon.models.layers import layer_dropout
from gluon.models.layers import layer_linear
from gluon.models.layers import activation_elu
from gluon.models.layers import activation_leakyrelu
from gluon.models.layers import activation_prelu
from gluon.models.layers import activation_relu
from gluon.models.layers import activation_selu
from gluon.models.layers import activation_sigmoid
from gluon.models.layers import activation_softplus
from gluon.models.layers import activation_softsign
from gluon.models.layers import activation_swish
from gluon.models.layers import activation_tanh

from gluon.models.params import set_model_name
from gluon.models.params import set_device
from gluon.models.params import set_pretrained
from gluon.models.params import set_freeze_base_network
from gluon.models.params import set_model_path

from gluon.models.common import set_parameter_requires_grad
from gluon.models.common import model_to_device
from gluon.models.common import print_grad_stats
from gluon.models.common import get_num_layers
from gluon.models.common import freeze_layers

from gluon.models.return_model import load_model
from gluon.models.return_model import setup_model
from gluon.models.return_model import debug_create_network

from gluon.models.features import CNNVisualizer
################################################################################




################################################################################
from gluon.optimizers.optimizers import sgd
from gluon.optimizers.optimizers import nesterov_sgd
from gluon.optimizers.optimizers import rmsprop
from gluon.optimizers.optimizers import momentum_rmsprop
from gluon.optimizers.optimizers import adam
from gluon.optimizers.optimizers import adagrad
from gluon.optimizers.optimizers import adadelta
from gluon.optimizers.optimizers import adamax
from gluon.optimizers.optimizers import nesterov_adam
from gluon.optimizers.optimizers import signum

from gluon.optimizers.retrieve_optimizer import retrieve_optimizer

from gluon.optimizers.return_optimizer import load_optimizer
################################################################################



################################################################################
from gluon.schedulers.schedulers import scheduler_fixed
from gluon.schedulers.schedulers import scheduler_step
from gluon.schedulers.schedulers import scheduler_multistep

from gluon.schedulers.retrieve_scheduler import retrieve_scheduler

from gluon.schedulers.return_scheduler import load_scheduler
################################################################################





################################################################################
from gluon.testing.process import process_single
from gluon.testing.process import process_multi
################################################################################




################################################################################
from gluon.training.params import set_num_epochs
from gluon.training.params import set_display_progress_realtime
from gluon.training.params import set_display_progress
from gluon.training.params import set_save_intermediate_models
from gluon.training.params import set_save_training_logs
from gluon.training.params import set_intermediate_model_prefix
################################################################################





################################################################################
from gluon.transforms.transforms import transform_random_resized_crop
from gluon.transforms.transforms import transform_center_crop
from gluon.transforms.transforms import transform_color_jitter
from gluon.transforms.transforms import transform_random_horizontal_flip
from gluon.transforms.transforms import transform_random_vertical_flip
from gluon.transforms.transforms import transform_random_lighting
from gluon.transforms.transforms import transform_resize
from gluon.transforms.transforms import transform_normalize

from gluon.transforms.return_transform import set_transform_trainval
from gluon.transforms.return_transform import set_transform_test

from gluon.transforms.retrieve_transform import retrieve_trainval_transforms
from gluon.transforms.retrieve_transform import retrieve_test_transforms

################################################################################
