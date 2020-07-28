import os
import sys

import psutil

from monk.gluon_prototype import prototype
from monk.compare_prototype import compare
from monk.pip_unit_tests.gluon.common import print_start
from monk.pip_unit_tests.gluon.common import print_status

import mxnet as mx
import numpy as np
from monk.gluon.losses.return_loss import load_loss


def test_block_inception_c(system_dict):
    forward = True;

    test = "test_block_inception_c";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");


            network = [];
            network.append(gtf.inception_c_block(channels_7x7=3, pool_type="avg"));
            network.append(gtf.inception_c_block(channels_7x7=3, pool_type="max"));
            gtf.Compile_Network(network, use_gpu=False);

            x = np.random.rand(1, 1, 64, 64);
            x = mx.nd.array(x);
            y = gtf.system_dict["local"]["model"].forward(x);          

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
