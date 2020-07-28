import os
import sys

import psutil

from monk.keras_prototype import prototype
from monk.compare_prototype import compare
from monk.pip_unit_tests.keras.common import print_start
from monk.pip_unit_tests.keras.common import print_status

import numpy as np
from monk.tf_keras_1.losses.return_loss import load_loss
from keras import backend as K

import tensorflow as tf
if(tf.__version__[0] == '2'):
    import tensorflow.compat.v1 as tf
    tf.enable_v2_behavior()


def test_loss_hinge(system_dict):
    forward = True;

    test = "test_loss_hinge";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");

            y = np.random.randn(1, 5);
            label = np.random.randn(1, 5);

            y = K.constant(y);
            label = K.constant(label);

            gtf.loss_hinge();
            load_loss(gtf.system_dict);
            loss_obj = gtf.system_dict["local"]["criterion"];
            loss_val = loss_obj(label, y);           

            system_dict["successful_tests"] += 1;
            print_status("Pass");

        except Exception as e:
            system_dict["failed_tests_exceptions"].append(e);
            system_dict["failed_tests_lists"].append(test);
            forward = False;
            print_status("Fail");
    else:
        system_dict["skipped_tests_lists"].append(test);
        print_status("Skipped");

    return system_dict
