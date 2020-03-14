import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype



##########################################################################################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");


gtf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=10);

gtf.Train();
##########################################################################################################################



# Press CTRL-C to interrupt training



################################################# Resume Training ########################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1", resume_train=True);
gtf.Train();
##########################################################################################################################