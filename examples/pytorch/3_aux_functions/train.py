import os
import sys
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype



ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");


ptf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18", freeze_base_network=True, num_epochs=2);




########################################################   Summary    #####################################################
ptf.Summary()
###########################################################################################################################





##################################################### EDA - Find Num images per class #####################################
ptf.EDA(show_img=True, save_img=True);
###########################################################################################################################





##################################################### EDA - Find Missing and corrupted images #####################################
ptf.EDA(check_missing=True, check_corrupt=True);
###########################################################################################################################





##################################################### Estimate Training Time #####################################
ptf.Estimate_Train_Time(num_epochs=50);
###########################################################################################################################