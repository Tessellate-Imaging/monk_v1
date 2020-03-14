import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype



gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");


gtf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=2);




########################################################   Summary    #####################################################
gtf.Summary()
###########################################################################################################################





##################################################### EDA - Find Num images per class #####################################
gtf.EDA(show_img=True, save_img=True);
###########################################################################################################################





##################################################### EDA - Find Missing and corrupted images #####################################
gtf.EDA(check_missing=True, check_corrupt=True);
###########################################################################################################################





##################################################### Estimate Training Time #####################################
gtf.Estimate_Train_Time(num_epochs=50);
###########################################################################################################################