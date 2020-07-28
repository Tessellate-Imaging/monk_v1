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



def test_initializer_he_uniform(system_dict):
    forward = True;

    test = "test_initializer_he_uniform";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");


            network = [];
            network.append(gtf.convolution(output_channels=16));
            network.append(gtf.batch_normalization());
            network.append(gtf.relu());
            network.append(gtf.convolution(output_channels=32));
            network.append(gtf.batch_normalization());
            network.append(gtf.relu());
            network.append(gtf.average_pooling(kernel_size=2));


            network.append(gtf.convolution(output_channels=64));
            network.append(gtf.batch_normalization());
            network.append(gtf.relu());
            network.append(gtf.convolution(output_channels=64));
            network.append(gtf.batch_normalization());
            network.append(gtf.relu());
            network.append(gtf.average_pooling(kernel_size=2));


            network.append(gtf.convolution(output_channels=128));
            network.append(gtf.batch_normalization());
            network.append(gtf.relu());
            network.append(gtf.convolution(output_channels=128));
            network.append(gtf.batch_normalization());
            network.append(gtf.relu());
            network.append(gtf.average_pooling(kernel_size=2));

            network.append(gtf.flatten());
            network.append(gtf.dropout(drop_probability=0.2));
            network.append(gtf.fully_connected(units=1024));
            network.append(gtf.dropout(drop_probability=0.2));
            network.append(gtf.fully_connected(units=2));
            network.append(gtf.softmax());

            gtf.Compile_Network(network, data_shape=(3, 32, 32), network_initializer="he_uniform");


            x = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
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
