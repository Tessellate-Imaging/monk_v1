import os
import sys
sys.path.append("../../../monk/");
import psutil

from keras_prototype import prototype



##########################################################################################################################
ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-1");


ktf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet50", freeze_base_network=True, num_epochs=10);

ktf.Train();
##########################################################################################################################



# Press CTRL-C to interrupt training



################################################# Resume Training ########################################################
ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-1", resume_train=True);
ktf.Train();
##########################################################################################################################