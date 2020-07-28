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
from monk.system.common import read_json
from monk.system.common import write_json
from monk.system.common import parse_csv
from monk.system.common import parse_csv2
from monk.system.common import parse_csv2_updated
from monk.system.common import save

from monk.system.summary import print_summary
################################################################################





################################################################################
from monk.tf_keras_1.datasets.params import set_input_size
from monk.tf_keras_1.datasets.params import set_batch_size
from monk.tf_keras_1.datasets.params import set_data_shuffle
from monk.tf_keras_1.datasets.params import set_num_processors
from monk.tf_keras_1.datasets.params import set_weighted_sampling

from monk.tf_keras_1.datasets.paths import set_dataset_train_path
from monk.tf_keras_1.datasets.paths import set_dataset_test_path
################################################################################






################################################################################
from monk.tf_keras_1.transforms.transforms import transform_color_jitter
from monk.tf_keras_1.transforms.transforms import transform_random_affine
from monk.tf_keras_1.transforms.transforms import transform_random_horizontal_flip
from monk.tf_keras_1.transforms.transforms import transform_random_rotation
from monk.tf_keras_1.transforms.transforms import transform_random_vertical_flip
from monk.tf_keras_1.transforms.transforms import transform_mean_subtraction
from monk.tf_keras_1.transforms.transforms import transform_normalize

from monk.tf_keras_1.transforms.return_transform import set_transform_trainval
from monk.tf_keras_1.transforms.return_transform import set_transform_test
from monk.tf_keras_1.transforms.return_transform import set_transform_estimate


from monk.tf_keras_1.transforms.retrieve_transform import retrieve_trainval_transforms
from monk.tf_keras_1.transforms.retrieve_transform import retrieve_test_transforms
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
from monk.tf_keras_1.models.layers import layer_dropout
from monk.tf_keras_1.models.layers import layer_linear
from monk.tf_keras_1.models.layers import layer_globalaveragepooling
from monk.tf_keras_1.models.layers import layer_flatten

from monk.tf_keras_1.models.layers import activation_leakyrelu
from monk.tf_keras_1.models.layers import activation_prelu
from monk.tf_keras_1.models.layers import activation_elu
from monk.tf_keras_1.models.layers import activation_threshold
from monk.tf_keras_1.models.layers import activation_softmax
from monk.tf_keras_1.models.layers import activation_relu
from monk.tf_keras_1.models.layers import activation_selu
from monk.tf_keras_1.models.layers import activation_softplus
from monk.tf_keras_1.models.layers import activation_softsign
from monk.tf_keras_1.models.layers import activation_tanh
from monk.tf_keras_1.models.layers import activation_sigmoid


from monk.tf_keras_1.models.params import set_model_name
from monk.tf_keras_1.models.params import set_device
from monk.tf_keras_1.models.params import set_pretrained
from monk.tf_keras_1.models.params import set_freeze_base_network
from monk.tf_keras_1.models.params import set_model_path
from monk.tf_keras_1.models.params import set_gpu_memory_fraction



from monk.tf_keras_1.models.common import set_parameter_requires_grad
from monk.tf_keras_1.models.common import print_grad_stats
from monk.tf_keras_1.models.common import get_num_layers
from monk.tf_keras_1.models.common import freeze_layers
from monk.tf_keras_1.models.common import get_num_trainable_layers
from monk.tf_keras_1.models.common import setup_device_environment

from monk.tf_keras_1.models.return_model import load_model
from monk.tf_keras_1.models.return_model import setup_model
from monk.tf_keras_1.models.return_model import debug_create_network

from monk.tf_keras_1.models.features import CNNVisualizer
################################################################################






################################################################################
from monk.tf_keras_1.schedulers.schedulers import scheduler_fixed
from monk.tf_keras_1.schedulers.schedulers import scheduler_step
from monk.tf_keras_1.schedulers.schedulers import scheduler_exponential
from monk.tf_keras_1.schedulers.schedulers import scheduler_plateau

from monk.tf_keras_1.schedulers.retrieve_scheduler import retrieve_scheduler

from monk.tf_keras_1.schedulers.return_scheduler import load_scheduler
################################################################################







################################################################################
from monk.tf_keras_1.optimizers.optimizers import adadelta
from monk.tf_keras_1.optimizers.optimizers import adagrad
from monk.tf_keras_1.optimizers.optimizers import adam
from monk.tf_keras_1.optimizers.optimizers import adamax
from monk.tf_keras_1.optimizers.optimizers import rmsprop
from monk.tf_keras_1.optimizers.optimizers import sgd
from monk.tf_keras_1.optimizers.optimizers import nesterov_sgd
from monk.tf_keras_1.optimizers.optimizers import nesterov_adam

from monk.tf_keras_1.optimizers.retrieve_optimizer import retrieve_optimizer

from monk.tf_keras_1.optimizers.return_optimizer import load_optimizer
################################################################################







################################################################################
from monk.tf_keras_1.losses.losses import l1
from monk.tf_keras_1.losses.losses import l2
from monk.tf_keras_1.losses.losses import crossentropy
from monk.tf_keras_1.losses.losses import binary_crossentropy
from monk.tf_keras_1.losses.losses import kldiv
from monk.tf_keras_1.losses.losses import hinge
from monk.tf_keras_1.losses.losses import squared_hinge

#from tf_keras_1.losses.losses import sparse_categorical_crossentropy
#from tf_keras_1.losses.losses import categorical_hinge
#from tf_keras_1.losses.losses import binary_crossentropy

from monk.tf_keras_1.losses.return_loss import load_loss

from monk.tf_keras_1.losses.retrieve_loss import retrieve_loss
################################################################################






################################################################################
from monk.tf_keras_1.training.params import set_num_epochs
from monk.tf_keras_1.training.params import set_display_progress_realtime
from monk.tf_keras_1.training.params import set_display_progress
from monk.tf_keras_1.training.params import set_save_intermediate_models
from monk.tf_keras_1.training.params import set_save_training_logs
from monk.tf_keras_1.training.params import set_intermediate_model_prefix


from monk.tf_keras_1.training.callbacks import TimeHistory
from monk.tf_keras_1.training.callbacks import MemoryHistory
################################################################################





################################################################################
from monk.tf_keras_1.testing.process import process_single
from monk.tf_keras_1.testing.process import process_multi
################################################################################
