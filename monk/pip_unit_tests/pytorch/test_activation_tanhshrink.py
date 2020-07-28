import os
import sys

import psutil

from monk.pytorch_prototype import prototype
from monk.compare_prototype import compare
from monk.pip_unit_tests.pytorch.common import print_start
from monk.pip_unit_tests.pytorch.common import print_status

import torch
import numpy as np
from monk.pytorch.losses.return_loss import load_loss


def test_activation_tanhshrink(system_dict):
    forward = True;

    test = "test_activation_tanhshrink";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");


            network = [];
            network.append(gtf.tanhshrink());
            gtf.Compile_Network(network, data_shape=(3, 64, 64), use_gpu=False);

            x = torch.randn(1, 3, 64, 64);
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
