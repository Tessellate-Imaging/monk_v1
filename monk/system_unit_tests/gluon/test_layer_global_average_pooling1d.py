import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype
from compare_prototype import compare
from common import print_start
from common import print_status

import mxnet as mx
import numpy as np
from gluon.losses.return_loss import load_loss


def test_layer_global_average_pooling1d(system_dict):
    forward = True;

    test = "test_layer_global_average_pooling1d";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");


            network = [];
            network.append(gtf.global_average_pooling1d());
            gtf.Compile_Network(network, use_gpu=False);

            x = np.random.rand(1, 64, 4);
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
