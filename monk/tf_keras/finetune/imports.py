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


stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras import backend as K
import tensorflow as tf
import keras.callbacks as krc
from tabulate import tabulate

################################################################################3
from system.common import read_json
from system.common import write_json
from system.common import parse_csv
from system.common import parse_csv2
from system.common import save

from system.summary import print_summary
################################################################################





################################################################################
from tf_keras.datasets.params import set_input_size
from tf_keras.datasets.params import set_batch_size
from tf_keras.datasets.params import set_data_shuffle
from tf_keras.datasets.params import set_num_processors
from tf_keras.datasets.params import set_weighted_sampling

from tf_keras.datasets.paths import set_dataset_train_path
from tf_keras.datasets.paths import set_dataset_test_path
################################################################################






################################################################################
from tf_keras.transforms.transforms import transform_color_jitter
from tf_keras.transforms.transforms import transform_random_affine
from tf_keras.transforms.transforms import transform_random_horizontal_flip
from tf_keras.transforms.transforms import transform_random_rotation
from tf_keras.transforms.transforms import transform_random_vertical_flip
from tf_keras.transforms.transforms import transform_mean_subtraction
from tf_keras.transforms.transforms import transform_normalize

from tf_keras.transforms.return_transform import set_transform_trainval
from tf_keras.transforms.return_transform import set_transform_test
from tf_keras.transforms.return_transform import set_transform_estimate


from tf_keras.transforms.retrieve_transform import retrieve_trainval_transforms
from tf_keras.transforms.retrieve_transform import retrieve_test_transforms
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
from tf_keras.models.layers import layer_dropout
from tf_keras.models.layers import layer_linear
from tf_keras.models.layers import layer_globalaveragepooling
from tf_keras.models.layers import layer_flatten

from tf_keras.models.layers import activation_leakyrelu
from tf_keras.models.layers import activation_prelu
from tf_keras.models.layers import activation_elu
from tf_keras.models.layers import activation_threshold
from tf_keras.models.layers import activation_softmax
from tf_keras.models.layers import activation_relu
from tf_keras.models.layers import activation_selu
from tf_keras.models.layers import activation_softplus
from tf_keras.models.layers import activation_softsign
from tf_keras.models.layers import activation_tanh
from tf_keras.models.layers import activation_sigmoid


from tf_keras.models.params import set_model_name
from tf_keras.models.params import set_device
from tf_keras.models.params import set_pretrained
from tf_keras.models.params import set_freeze_base_network
from tf_keras.models.params import set_model_path
from tf_keras.models.params import set_gpu_memory_fraction



from tf_keras.models.common import set_parameter_requires_grad
from tf_keras.models.common import print_grad_stats
from tf_keras.models.common import get_num_layers
from tf_keras.models.common import freeze_layers
from tf_keras.models.common import get_num_trainable_layers
from tf_keras.models.common import setup_device_environment

from tf_keras.models.return_model import load_model
from tf_keras.models.return_model import setup_model
################################################################################






################################################################################
from tf_keras.schedulers.schedulers import scheduler_fixed
from tf_keras.schedulers.schedulers import scheduler_step
from tf_keras.schedulers.schedulers import scheduler_exponential
from tf_keras.schedulers.schedulers import scheduler_plateau

from tf_keras.schedulers.retrieve_scheduler import retrieve_scheduler

from tf_keras.schedulers.return_scheduler import load_scheduler
################################################################################







################################################################################
from tf_keras.optimizers.optimizers import adadelta
from tf_keras.optimizers.optimizers import adagrad
from tf_keras.optimizers.optimizers import adam
from tf_keras.optimizers.optimizers import adamax
from tf_keras.optimizers.optimizers import rmsprop
from tf_keras.optimizers.optimizers import sgd
from tf_keras.optimizers.optimizers import nesterov_sgd
from tf_keras.optimizers.optimizers import nesterov_adam

from tf_keras.optimizers.retrieve_optimizer import retrieve_optimizer

from tf_keras.optimizers.return_optimizer import load_optimizer
################################################################################







################################################################################
from tf_keras.losses.losses import categorical_crossentropy
from tf_keras.losses.losses import sparse_categorical_crossentropy
from tf_keras.losses.losses import categorical_hinge
from tf_keras.losses.losses import binary_crossentropy

from tf_keras.losses.return_loss import load_loss

from tf_keras.losses.retrieve_loss import retrieve_loss
################################################################################






################################################################################
from tf_keras.training.params import set_num_epochs
from tf_keras.training.params import set_display_progress_realtime
from tf_keras.training.params import set_display_progress
from tf_keras.training.params import set_save_intermediate_models
from tf_keras.training.params import set_save_training_logs
from tf_keras.training.params import set_intermediate_model_prefix


from tf_keras.training.callbacks import TimeHistory
from tf_keras.training.callbacks import MemoryHistory
################################################################################





################################################################################
from tf_keras.testing.process import process_single
################################################################################
