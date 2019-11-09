import os
import sys
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype



##########################################################################################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");


ptf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18", freeze_base_network=True, num_epochs=10);

ptf.Train();
##########################################################################################################################



# Press CTRL-C to interrupt training



################################################# Resume Training ########################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1", resume_train=True);
ptf.Train();
##########################################################################################################################