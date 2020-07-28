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


def test_loss_binary_crossentropy(system_dict):
    forward = True;

    test = "test_loss_binary_crossentropy";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");

            label = torch.empty((1, 5)).random_(2);

            y = torch.randn(1, 5);
            m = torch.nn.Sigmoid();
            y = m(y);

            gtf.loss_binary_crossentropy();
            load_loss(gtf.system_dict);
            loss_obj = gtf.system_dict["local"]["criterion"];
            loss_val = loss_obj(y, label);           

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
