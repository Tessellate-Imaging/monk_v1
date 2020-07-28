import os
import sys
sys.path.append("../../../../monk_v1/");
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype
from compare_prototype import compare
from common import print_start
from common import print_status

import torch
import numpy as np
from pytorch.losses.return_loss import load_loss


def test_layer_concatenate(system_dict):
    forward = True;

    test = "test_layer_concatenate";
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
            network.append(gtf.max_pooling());



            subnetwork = [];
            branch1 = [];
            branch1.append(gtf.convolution(output_channels=16));
            branch1.append(gtf.batch_normalization());
            branch1.append(gtf.convolution(output_channels=16));
            branch1.append(gtf.batch_normalization());

            branch2 = [];
            branch2.append(gtf.convolution(output_channels=16));
            branch2.append(gtf.batch_normalization());

            branch3 = [];
            branch3.append(gtf.identity())

            subnetwork.append(branch1);
            subnetwork.append(branch2);
            subnetwork.append(branch3);
            subnetwork.append(gtf.concatenate())


            network.append(subnetwork);



            network.append(gtf.convolution(output_channels=16));
            network.append(gtf.batch_normalization());
            network.append(gtf.relu());
            network.append(gtf.max_pooling());

            network.append(gtf.flatten());
            network.append(gtf.fully_connected(units=1024));
            network.append(gtf.dropout(drop_probability=0.2));
            network.append(gtf.fully_connected(units=2));


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
