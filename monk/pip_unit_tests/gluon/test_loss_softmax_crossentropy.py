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


def test_loss_softmax_crossentropy(system_dict):
    forward = True;

    test = "test_loss_softmax_crossentropy";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            gtf = prototype(verbose=0);
            gtf.Prototype("sample-project-1", "sample-experiment-1");

            label = np.random.rand(1, 1);
            label = mx.nd.array(label);

            y = np.random.rand(1, 5);
            y = mx.nd.array(y);

            gtf.loss_softmax_crossentropy();
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
