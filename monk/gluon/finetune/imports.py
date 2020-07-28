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
from monk.system.common import read_json
from monk.system.common import write_json
from monk.system.common import parse_csv
from monk.system.common import parse_csv_updated
from monk.system.common import save

from monk.system.summary import print_summary
################################################################################




################################################################################
from monk.gluon.datasets.class_imbalance import balance_class_weights

from monk.gluon.datasets.params import set_input_size
from monk.gluon.datasets.params import set_batch_size
from monk.gluon.datasets.params import set_data_shuffle
from monk.gluon.datasets.params import set_num_processors
from monk.gluon.datasets.params import set_weighted_sampling

from monk.gluon.datasets.csv_dataset import DatasetCustom
from monk.gluon.datasets.csv_dataset import DatasetCustomMultiLabel
from monk.gluon.datasets.paths import set_dataset_train_path
from monk.gluon.datasets.paths import set_dataset_test_path
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
from monk.gluon.losses.losses import softmax_crossentropy
from monk.gluon.losses.losses import crossentropy
from monk.gluon.losses.losses import sigmoid_binary_crossentropy
from monk.gluon.losses.losses import binary_crossentropy
from monk.gluon.losses.losses import poisson_nll
from monk.gluon.losses.losses import l1
from monk.gluon.losses.losses import l2
from monk.gluon.losses.losses import kldiv
from monk.gluon.losses.losses import huber
from monk.gluon.losses.losses import hinge
from monk.gluon.losses.losses import squared_hinge


from monk.gluon.losses.return_loss import load_loss

from monk.gluon.losses.retrieve_loss import retrieve_loss
################################################################################




################################################################################
from monk.gluon.models.layers import layer_dropout
from monk.gluon.models.layers import layer_linear
from monk.gluon.models.layers import activation_elu
from monk.gluon.models.layers import activation_leakyrelu
from monk.gluon.models.layers import activation_prelu
from monk.gluon.models.layers import activation_relu
from monk.gluon.models.layers import activation_selu
from monk.gluon.models.layers import activation_sigmoid
from monk.gluon.models.layers import activation_softplus
from monk.gluon.models.layers import activation_softsign
from monk.gluon.models.layers import activation_swish
from monk.gluon.models.layers import activation_tanh

from monk.gluon.models.params import set_model_name
from monk.gluon.models.params import set_device
from monk.gluon.models.params import set_pretrained
from monk.gluon.models.params import set_freeze_base_network
from monk.gluon.models.params import set_model_path

from monk.gluon.models.common import set_parameter_requires_grad
from monk.gluon.models.common import model_to_device
from monk.gluon.models.common import print_grad_stats
from monk.gluon.models.common import get_num_layers
from monk.gluon.models.common import freeze_layers

from monk.gluon.models.return_model import load_model
from monk.gluon.models.return_model import setup_model
from monk.gluon.models.return_model import debug_create_network

from monk.gluon.models.features import CNNVisualizer
################################################################################




################################################################################
from monk.gluon.optimizers.optimizers import sgd
from monk.gluon.optimizers.optimizers import nesterov_sgd
from monk.gluon.optimizers.optimizers import rmsprop
from monk.gluon.optimizers.optimizers import momentum_rmsprop
from monk.gluon.optimizers.optimizers import adam
from monk.gluon.optimizers.optimizers import adagrad
from monk.gluon.optimizers.optimizers import adadelta
from monk.gluon.optimizers.optimizers import adamax
from monk.gluon.optimizers.optimizers import nesterov_adam
from monk.gluon.optimizers.optimizers import signum

from monk.gluon.optimizers.retrieve_optimizer import retrieve_optimizer

from monk.gluon.optimizers.return_optimizer import load_optimizer
################################################################################



################################################################################
from monk.gluon.schedulers.schedulers import scheduler_fixed
from monk.gluon.schedulers.schedulers import scheduler_step
from monk.gluon.schedulers.schedulers import scheduler_multistep

from monk.gluon.schedulers.retrieve_scheduler import retrieve_scheduler

from monk.gluon.schedulers.return_scheduler import load_scheduler
################################################################################





################################################################################
from monk.gluon.testing.process import process_single
from monk.gluon.testing.process import process_multi
################################################################################




################################################################################
from monk.gluon.training.params import set_num_epochs
from monk.gluon.training.params import set_display_progress_realtime
from monk.gluon.training.params import set_display_progress
from monk.gluon.training.params import set_save_intermediate_models
from monk.gluon.training.params import set_save_training_logs
from monk.gluon.training.params import set_intermediate_model_prefix
################################################################################





################################################################################
from monk.gluon.transforms.transforms import transform_random_resized_crop
from monk.gluon.transforms.transforms import transform_center_crop
from monk.gluon.transforms.transforms import transform_color_jitter
from monk.gluon.transforms.transforms import transform_random_horizontal_flip
from monk.gluon.transforms.transforms import transform_random_vertical_flip
from monk.gluon.transforms.transforms import transform_random_lighting
from monk.gluon.transforms.transforms import transform_resize
from monk.gluon.transforms.transforms import transform_normalize

from monk.gluon.transforms.return_transform import set_transform_trainval
from monk.gluon.transforms.return_transform import set_transform_test

from monk.gluon.transforms.retrieve_transform import retrieve_trainval_transforms
from monk.gluon.transforms.retrieve_transform import retrieve_test_transforms

################################################################################
