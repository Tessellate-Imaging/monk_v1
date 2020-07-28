import os
import sys

import psutil

from monk.keras_prototype import prototype
from monk.compare_prototype import compare
from monk.pip_unit_tests.keras.common import print_start
from monk.pip_unit_tests.keras.common import print_status

import tensorflow as tf
if(tf.__version__[0] == '2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np



def test_layer_convolution3d(system_dict):
    forward = True;

    test = "test_layer_convolution3d";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");


            network = [];
            network.append(gtf.convolution3d(output_channels=3, kernel_size=3));
            gtf.Compile_Network(network, data_shape=(3, 10, 32, 32), use_gpu=False);

            x = tf.placeholder(tf.float32, shape=(1, 10, 32, 32, 3))
            y = gtf.system_dict["local"]["model"](x);          

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
